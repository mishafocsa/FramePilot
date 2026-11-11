from dataclasses import dataclass, replace

import cv2
import numpy as np

from .automation import Automation
from .calibration import Calibration, compensate_rect, receiver_to_output, rotate_image
from .composition_quality import composition_report
from .foreground import Background, PersonSegmenter, composite
from .framing import WHOLE, AutoFraming, FramingRule
from .geometry import Placement, Rect, fit, smooth
from .masks import FrameMask, MaskRenderer
from .object_geometry import QuadSelection, quad_bounds, rectangle_quad, region_quad, valid_quad
from .objects import ObjectTracker
from .vision import FaceDetector, FaceTracker, PeopleDetector


@dataclass(frozen=True)
class SceneSlot:
    identifier: str
    target: Rect
    shape: str = "rectangle"
    subject_mode: str = "face"
    padding: float = 0.25
    fill: bool = False
    eye_level: float | None = None
    max_zoom: float = 6.0
    subject_region: Rect = Rect(0, 0, 1, 1)
    framing: FramingRule = FramingRule()
    whole_fit: bool = False  # Resolved export state, not persisted in user profiles.
    frame_mask: FrameMask = FrameMask()
    framing_zoom: float = 1.0
    surface_lock: bool = False

    def record(self):
        return {
            "id": self.identifier,
            "target": [self.target.x, self.target.y, self.target.w, self.target.h],
            "shape": self.shape,
            "subject_mode": self.subject_mode,
            "padding": self.padding,
            "fill": self.fill,
            "eye_level": self.eye_level,
            "max_zoom": self.max_zoom,
            "subject_region": [
                self.subject_region.x,
                self.subject_region.y,
                self.subject_region.w,
                self.subject_region.h,
            ],
            "framing": self.framing.record(),
            "frame_mask": self.frame_mask.record(),
            "framing_zoom": self.framing_zoom,
            "surface_lock": self.surface_lock,
        }

    @classmethod
    def from_record(cls, value):
        return cls(
            value["id"],
            Rect(*value["target"]),
            value["shape"],
            value["subject_mode"],
            value["padding"],
            value["fill"],
            value["eye_level"],
            value["max_zoom"],
            Rect(*value.get("subject_region", [0, 0, 1, 1])),
            FramingRule.from_record(value.get("framing")),
            frame_mask=FrameMask.from_record(value.get("frame_mask")),
            framing_zoom=value.get("framing_zoom", 1.0),
            surface_lock=value.get("surface_lock", False),
        )


@dataclass(frozen=True)
class Settings:
    width: int = 1280
    height: int = 720
    target: Rect = Rect(0.3, 0.15, 0.4, 0.7)
    shape: str = "rectangle"
    padding: float = 0.25
    response: float = 0.35
    fill: bool = False
    loss: str = "freeze"
    # A camera slot already occupies the detected rectangle in the receiving app.
    # In that case output must fill the camera image rather than apply the offset twice.
    mapping: str = "camera"
    calibration: Calibration = Calibration()
    subject_mode: str = "face"
    eye_level: float | None = None
    max_zoom: float = 6.0
    slots: tuple[SceneSlot, ...] = ()
    active_slot: str = ""
    remove_background: bool = False
    segmentation_quality: str = "balanced"
    background_color: str = "#000000"
    background_image: str = ""
    subject_region: Rect = Rect(0, 0, 1, 1)
    motion_mode: str = "smooth"
    framing: FramingRule = FramingRule()
    whole_fit: bool = False
    frame_mask: FrameMask = FrameMask()
    framing_zoom: float = 1.0
    automation: Automation = Automation()
    automation_slot: str = "single"
    surface_lock: bool = False


class Renderer:
    def __init__(self) -> None:
        self.background = Background()
        self.masks = MaskRenderer()
        self.reset()

    def reset(self, preserve_auto=False) -> None:
        if not preserve_auto:
            self.auto_framing = AutoFraming()
        self.last_framing = {}
        self.placement: Placement | None = None
        self.previous: np.ndarray | None = None
        self.time: float | None = None
        self.size: tuple[int, int] | None = None
        self.layer = None
        self.coverage = None
        self.last_surface_quad = None
        self.last_report = {}

    def render(
        self,
        image: np.ndarray,
        face: Rect | None,
        settings: Settings,
        now: float,
        snap: bool = False,
        eyes: tuple[float, float] | None = None,
        foreground: np.ndarray | None = None,
        layer_only: bool = False,
        object_quad=None,
        source_opacity=None,
    ) -> np.ndarray:
        size = settings.width, settings.height
        if self.size != size:
            self.reset(preserve_auto=True)
            self.size = size
        dt = now - self.time if self.time is not None else 1 / 30
        self.time = now
        canvas = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
        backdrop = None if layer_only else self.background.frame(settings)
        ih, iw = image.shape[:2]
        self.last_surface_quad = None
        locked = settings.surface_lock and settings.subject_mode == "object"
        self.last_report = {
            "state": "unconfirmed",
            "warnings": ["Цель не подтверждена"],
            "scope": "Geometric framing evidence",
        }
        if face is not None and locked and not valid_quad(object_quad):
            face = None
            self.last_report = {
                "state": "surface_unavailable",
                "warnings": ["Нет геометрии поверхности: подтвердите цель или повторите анализ"],
                "scope": "Geometric framing evidence",
            }
        if face is None:
            # Freeze does not advance composition while there is no confirmed target.
            self.last_framing = self.auto_framing.status(settings.framing, settings.target.area)
            self.last_framing["transitioning"] = False
            if settings.loss == "freeze" and self.previous is not None:
                return self.layer if layer_only else composite(self.layer, self.coverage, backdrop)
            self.coverage = np.zeros(canvas.shape[:2], np.float32)
            if settings.loss == "original":
                destination = (
                    settings.target.clipped().pixels(*size)
                    if settings.mapping == "canvas"
                    else Rect(0, 0, settings.width, settings.height)
                )
                scale = min(destination.w / iw, destination.h / ih)
                rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
                x = round(destination.x + (destination.w - rw) / 2)
                y = round(destination.y + (destination.h - rh) / 2)
                if source_opacity is not None:
                    alpha = cv2.resize(source_opacity, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    pixels = cv2.resize(
                        image.astype(np.float32) * source_opacity[..., None],
                        (rw, rh),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    canvas[y : y + rh, x : x + rw] = np.clip(
                        pixels / np.maximum(alpha[..., None], 1e-6), 0, 255
                    ).astype(np.uint8)
                    self.coverage[y : y + rh, x : x + rw] = alpha
                else:
                    canvas[y : y + rh, x : x + rw] = cv2.resize(image, (rw, rh))
                    self.coverage[y : y + rh, x : x + rw] = 1
                self.coverage *= self.masks.alpha(size, destination, settings.shape, settings.frame_mask)
            self.layer = canvas
            self.previous = canvas if layer_only else composite(canvas, self.coverage, backdrop)
            return self.previous.copy()
        calibration = settings.calibration if settings.mapping == "camera" else Calibration()
        target = (calibration.crop if settings.mapping == "camera" else settings.target).clipped()
        if target.area <= 0:
            raise ValueError("Рамка должна иметь положительный размер")
        destination = target.pixels(*size)
        region = self.auto_framing.update(
            settings.subject_region, settings.target.area, settings.framing, now
        )
        self.last_framing = self.auto_framing.status(settings.framing, settings.target.area)
        whole_auto = settings.whole_fit or (settings.framing.enabled and self.auto_framing.whole is True)
        transitioning = settings.framing.enabled and self.auto_framing.active
        adaptive = settings.framing.enabled and settings.subject_region != WHOLE
        fullness = (
            max(
                0.0,
                min(
                    1.0,
                    (region.area - settings.subject_region.area)
                    / max(1 - settings.subject_region.area, 1e-9),
                ),
            )
            if adaptive
            else 0.0
        )
        if settings.whole_fit:
            fullness = 1.0
        partial = region != WHOLE
        focus = face.region(region).clipped()
        surface = region_quad(object_quad, region) if locked else None
        if surface is not None:
            self.last_surface_quad = [[float(x), float(y)] for x, y in surface]
            focus = quad_bounds(surface)
        if calibration.mirror or calibration.rotation:
            image = rotate_image(image, -calibration.rotation)
            if calibration.mirror:
                image = cv2.flip(image, 1)
            if foreground is not None:
                foreground = rotate_image(foreground, -calibration.rotation)
                if calibration.mirror:
                    foreground = cv2.flip(foreground, 1)
            if source_opacity is not None:
                source_opacity = rotate_image(source_opacity, -calibration.rotation)
                if calibration.mirror:
                    source_opacity = cv2.flip(source_opacity, 1)
            face = compensate_rect(face, calibration)
            focus = compensate_rect(focus, calibration)
            eyes = receiver_to_output(eyes, calibration) if eyes is not None else None
            if surface is not None:
                surface = np.float32([receiver_to_output(tuple(point), calibration) for point in surface])
            ih, iw = image.shape[:2]
        subject = focus.pixels(iw, ih)
        round_mask = settings.frame_mask.kind in ("ellipse", "circle") or (
            settings.frame_mask.kind == "auto" and settings.shape == "ellipse"
        )
        fit_fraction = 1 - 0.3 * fullness if round_mask else 1
        fit_area = (
            Rect(
                destination.center[0] - destination.w * fit_fraction / 2,
                destination.center[1] - destination.h * fit_fraction / 2,
                destination.w * fit_fraction,
                destination.h * fit_fraction,
            )
            if fit_fraction != 1
            else destination
        )
        if settings.frame_mask.kind == "circle" and (not settings.fill or whole_auto):
            edge = min(destination.w, destination.h) * (0.7 if not settings.fill else fit_fraction)
            fit_area = Rect(destination.center[0] - edge / 2, destination.center[1] - edge / 2, edge, edge)
        fit_scale = fit(subject, fit_area, settings.padding, False).scale
        fill_scale = fit(subject, fit_area, settings.padding, settings.fill).scale
        scale = fill_scale * (1 - fullness) + fit_scale * fullness
        anchor = subject.center
        target_anchor = destination.center
        if (
            eyes is not None
            and settings.eye_level is not None
            and not partial
            and not whole_auto
            and not adaptive
        ):
            anchor = eyes[0] * iw, eyes[1] * ih
            anchor_u, anchor_v = receiver_to_output((0.5, settings.eye_level), calibration)
            target_anchor = destination.x + destination.w * anchor_u, destination.y + destination.h * anchor_v
            if not settings.fill:
                left = anchor[0] - (subject.x - subject.w * settings.padding)
                right = subject.x + subject.w * (1 + settings.padding) - anchor[0]
                top = anchor[1] - (subject.y - subject.h * settings.padding)
                bottom = subject.y + subject.h * (1 + settings.padding) - anchor[1]
                scale = min(
                    scale,
                    (target_anchor[0] - destination.x) / max(left, 1),
                    (destination.x + destination.w - target_anchor[0]) / max(right, 1),
                    (target_anchor[1] - destination.y) / max(top, 1),
                    (destination.y + destination.h - target_anchor[1]) / max(bottom, 1),
                )
        base_scale = min(settings.width / iw, settings.height / ih)
        zoom = calibration.zoom * (1 - fullness) + min(1, calibration.zoom) * fullness
        scale = min(
            scale
            * zoom
            * (settings.framing_zoom * (1 - fullness) + min(1, settings.framing_zoom) * fullness),
            base_scale * settings.max_zoom,
        )
        offset_x, offset_y = calibration.offset_x * (1 - fullness), calibration.offset_y * (1 - fullness)
        offset_u, offset_v = receiver_to_output((0.5 + offset_x, 0.5 + offset_y), calibration)
        goal = Placement(
            scale,
            target_anchor[0] + (offset_u - 0.5) * destination.w - anchor[0] * scale,
            target_anchor[1] + (offset_v - 0.5) * destination.h - anchor[1] * scale,
        )
        if snap or transitioning:
            self.placement = None
        responsive = settings.motion_mode == "responsive"
        self.placement = smooth(
            self.placement,
            goal,
            dt,
            0.015 if responsive else settings.response,
            speed=max(size) * (32 if responsive else 0.8),
            zoom_rate=12 if responsive else 1.4,
        )
        p = self.placement
        if whole_auto and not transitioning:
            # Lag must never crop a confirmed whole target, even at zero padding.
            safe_scale = min(p.scale, goal.scale)
            p = Placement(
                safe_scale,
                min(
                    max(p.x, fit_area.x - subject.x * safe_scale),
                    fit_area.x + fit_area.w - (subject.x + subject.w) * safe_scale,
                ),
                min(
                    max(p.y, fit_area.y - subject.y * safe_scale),
                    fit_area.y + fit_area.h - (subject.y + subject.h) * safe_scale,
                ),
            )
            self.placement = p
        matrix = np.array([[p.scale, 0, p.x], [0, p.scale, p.y]], dtype=np.float32)
        source_alpha = foreground if foreground is not None else np.ones((ih, iw), np.float32)
        if source_opacity is not None:
            source_alpha = source_alpha * source_opacity
        focus_polygon = (
            surface * np.float32([iw, ih])
            if surface is not None
            else rectangle_quad(focus) * np.float32([iw, ih])
        )
        if surface is not None:
            alpha = np.zeros((ih, iw), np.uint8)
            cv2.fillConvexPoly(alpha, np.round(focus_polygon).astype(np.int32), 255, lineType=cv2.LINE_AA)
            source_alpha = source_alpha * (alpha.astype(np.float32) / 255)
        if partial or locked:
            # Crop before resampling, so the excluded half cannot bleed into the output.
            x1, y1 = max(0, round(subject.x)), max(0, round(subject.y))
            x2, y2 = (
                min(iw, max(x1 + 1, round(subject.x + subject.w))),
                min(ih, max(y1 + 1, round(subject.y + subject.h))),
            )
            image = image[y1:y2, x1:x2]
            source_alpha = source_alpha[y1:y2, x1:x2]
            matrix[0, 2] += x1 * p.scale
            matrix[1, 2] += y1 * p.scale
        transformed_alpha = cv2.warpAffine(
            source_alpha, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        if locked or source_opacity is not None:
            premultiplied = cv2.warpAffine(
                image.astype(np.float32) * source_alpha[..., None],
                matrix,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            transformed = np.clip(
                premultiplied / np.maximum(transformed_alpha[..., None], 1e-6), 0, 255
            ).astype(np.uint8)
        else:
            transformed = cv2.warpAffine(
                image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
        x, y = round(destination.x), round(destination.y)
        x2 = min(settings.width, round(destination.x + destination.w))
        y2 = min(settings.height, round(destination.y + destination.h))
        canvas[y:y2, x:x2] = transformed[y:y2, x:x2]
        self.coverage = np.zeros(canvas.shape[:2], np.float32)
        self.coverage[y:y2, x:x2] = transformed_alpha[y:y2, x:x2]
        destination_mask = self.masks.alpha(size, destination, settings.shape, settings.frame_mask)
        self.coverage *= destination_mask
        self.last_report = composition_report(focus_polygon, p, destination_mask, (iw, ih))
        self.last_report["surface_locked"] = bool(locked)
        self.layer = canvas
        self.previous = canvas if layer_only else composite(canvas, self.coverage, backdrop)
        return self.previous.copy()


class Pipeline:
    def __init__(self) -> None:
        self.detector = FaceDetector()
        self.tracker = FaceTracker(self.detector)
        self.object_tracker = ObjectTracker()
        self.person_tracker = FaceTracker(PeopleDetector(self.detector))
        self.upper_tracker = FaceTracker(PeopleDetector(self.detector, upper=True))
        self.renderer = Renderer()
        self.last_face = None
        self.last_faces = []
        self.eye_ratio = None
        self.last_eyes = None
        self.mode = None
        self.count = 0
        self.children = {}
        self.last_tracks = {}
        self.last_recovery = []
        self.last_quality = {}
        self.last_framing = {}
        self.segmenter = None
        self.last_object_quad = None
        self.mask_cache = None
        self.mask_key = None
        self.mask_tick = 0
        self.background = Background()
        self.applied_cue = None

    def reset(self) -> None:
        self.tracker.reset()
        self.object_tracker.reset()
        self.person_tracker.reset()
        self.upper_tracker.reset()
        self.renderer.reset()
        self.last_face = None
        self.last_faces = []
        self.eye_ratio = None
        self.last_eyes = None
        self.count = 0
        self.children = {}
        self.last_tracks = {}
        self.last_recovery = []
        self.last_quality = {}
        self.last_framing = {}
        self.mask_cache = None
        self.mask_key = None
        self.mask_tick = 0
        self.applied_cue = None
        self.last_object_quad = None

    def foreground(self, image, settings, repaint):
        if not settings.remove_background:
            return None
        key = image.shape[:2], settings.segmentation_quality
        if self.segmenter is None:
            self.segmenter = PersonSegmenter()
        if (
            self.mask_cache is None
            or self.mask_key != key
            or (not repaint and (settings.segmentation_quality == "accurate" or self.mask_tick % 2 == 0))
        ):
            self.mask_cache = self.segmenter.mask(image, settings.segmentation_quality)
            self.mask_key = key
        if not repaint:
            self.mask_tick += 1
        return self.mask_cache

    def process(
        self,
        image: np.ndarray,
        now: float,
        settings: Settings,
        click: tuple[float, float] | None,
        reset: bool,
        repaint: bool = False,
        mask_override: np.ndarray | None = None,
        layer_only: bool = False,
        render_time: float | None = None,
        source_opacity: np.ndarray | None = None,
        still: bool = False,
    ):
        if not settings.remove_background:
            self.mask_cache = None
        if settings.slots:
            if reset:
                self.reset()
            active = settings.active_slot or settings.slots[0].identifier
            ids = {slot.identifier for slot in settings.slots}
            self.children = {key: value for key, value in self.children.items() if key in ids}
            result = self.background.frame(settings)
            mask = (
                self.foreground(image, settings, repaint)
                if any(slot.subject_mode != "object" for slot in settings.slots)
                else None
            )
            self.last_tracks = {}
            for slot in settings.slots:
                if slot.identifier not in self.children:
                    self.children[slot.identifier] = Pipeline()
                child = self.children[slot.identifier]
                configuration = replace(
                    settings,
                    slots=(),
                    active_slot="",
                    mapping="canvas",
                    target=slot.target,
                    shape=slot.shape,
                    subject_mode=slot.subject_mode,
                    padding=slot.padding,
                    fill=slot.fill,
                    eye_level=slot.eye_level,
                    max_zoom=slot.max_zoom,
                    subject_region=slot.subject_region,
                    framing=slot.framing,
                    whole_fit=slot.whole_fit,
                    frame_mask=slot.frame_mask,
                    framing_zoom=slot.framing_zoom,
                    surface_lock=slot.surface_lock,
                    automation_slot=slot.identifier,
                    remove_background=settings.remove_background and slot.subject_mode != "object",
                )
                selection = (
                    click.get(slot.identifier)
                    if isinstance(click, dict)
                    else (click if active == slot.identifier else None)
                )
                child_reset = reset or (isinstance(click, dict) and click.get("_reset") == slot.identifier)
                output, subject, faces = child.process(
                    image,
                    now,
                    configuration,
                    selection,
                    child_reset,
                    repaint,
                    mask_override=mask,
                    layer_only=True,
                    render_time=render_time,
                    source_opacity=source_opacity,
                    still=still,
                )
                if child.renderer.layer is not None:
                    result = composite(child.renderer.layer, child.renderer.coverage, result)
                self.last_tracks[slot.identifier] = {
                    "subject": subject,
                    "mode": slot.subject_mode,
                    "framing": child.last_framing,
                    "tracking": child.last_quality,
                    "composition": child.renderer.last_report,
                    "surface": child.renderer.last_surface_quad,
                    "quad": child.last_object_quad,
                }
                if slot.identifier == active:
                    self.last_face, self.last_faces, self.last_eyes = subject, faces, child.last_eyes
                    self.last_recovery = child.last_recovery
                    self.last_quality = child.last_quality
                    self.last_framing = child.last_framing
            return result, self.last_face, self.last_faces
        mode_changed = self.mode != settings.subject_mode
        if reset or mode_changed:
            self.reset()
            self.mode = settings.subject_mode
        settings = settings.automation.settings_at(settings, now)
        cue = settings.automation.cue(settings.automation_slot, now) if settings.automation.enabled else None
        if cue != self.applied_cue:
            if cue is not None and abs(now - cue.time) < 0.15 and cue.subject is not None:
                click = (
                    QuadSelection(cue.quad)
                    if settings.subject_mode == "object" and cue.quad is not None
                    else cue.subject
                    if settings.subject_mode == "object"
                    else cue.subject.center
                )
            self.applied_cue = cue
        cached, sample = settings.automation.cached(settings.automation_slot, settings.subject_mode, now)
        h, w = image.shape[:2]
        scale = min(1, 960 / max(w, h))
        analysis = (
            cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else image
        )
        static_object = still and settings.subject_mode == "object"
        if static_object:
            # A still image has no motion to estimate. Even textureless regions are valid.
            if isinstance(click, (Rect, QuadSelection)):
                self.last_face = click.clipped() if isinstance(click, Rect) else quad_bounds(click.quad)
                quad = rectangle_quad(self.last_face) if isinstance(click, Rect) else click.quad
                self.last_object_quad = [[float(v) for v in p] for p in quad]
            self.last_faces = [self.last_face] if self.last_face is not None else []
            self.last_recovery = []
            self.last_quality = {"state": "static", "basis": "выбранная область фото"}
        elif cached:
            self.last_face = sample.subject
            self.last_faces = [sample.subject] if sample.subject else []
            self.last_eyes = sample.eyes
            self.last_recovery = []
            self.last_quality = (
                {
                    "state": sample.state,
                    "basis": "предварительный анализ",
                    "scope": "Cached tracker evidence; not ground-truth accuracy",
                }
                if settings.subject_mode == "object"
                else {}
            )
        elif not repaint or click is not None or reset or mode_changed:
            if click is not None:
                self.eye_ratio = None
            tracker = {
                "object": self.object_tracker,
                "person": self.person_tracker,
                "upper": self.upper_tracker,
            }.get(settings.subject_mode, self.tracker)
            self.last_face, self.last_faces = tracker.update(analysis, now, click)
            self.count += 1
            self.last_recovery = getattr(tracker, "proposals", [])
            self.last_quality = tracker.diagnostics() if settings.subject_mode == "object" else {}
        face = self.last_face
        object_quad = (
            self.last_object_quad
            if static_object
            else sample.quad
            if cached
            else self.object_tracker.quad
            if settings.subject_mode == "object" and face is not None
            else None
        )
        self.last_object_quad = (
            [[float(v) for v in p] for p in object_quad] if object_quad is not None else None
        )
        if cached:
            pass
        elif settings.subject_mode == "face" and settings.eye_level is not None and face is not None:
            if self.eye_ratio is None or self.count % 6 == 1 or click is not None:
                eyes = self.detector.eyes(analysis, face)
                if eyes is not None:
                    self.eye_ratio = ((eyes[0] - face.x) / face.w, (eyes[1] - face.y) / face.h)
            self.last_eyes = (
                (face.x + self.eye_ratio[0] * face.w, face.y + self.eye_ratio[1] * face.h)
                if self.eye_ratio
                else None
            )
        else:
            self.last_eyes = None
        foreground = None
        if settings.remove_background and settings.subject_mode != "object":
            foreground = (
                mask_override if mask_override is not None else self.foreground(image, settings, repaint)
            )
        output = self.renderer.render(
            image,
            face,
            settings,
            now if render_time is None else render_time,
            snap=repaint,
            eyes=self.last_eyes,
            foreground=foreground,
            layer_only=layer_only,
            object_quad=object_quad,
            source_opacity=source_opacity,
        )
        self.last_framing = self.renderer.last_framing
        self.last_tracks = {
            "single": {
                "subject": face,
                "mode": settings.subject_mode,
                "framing": self.last_framing,
                "tracking": self.last_quality,
                "composition": self.renderer.last_report,
                "surface": self.renderer.last_surface_quad,
                "quad": self.last_object_quad,
            }
        }
        return output, self.last_face, self.last_faces
