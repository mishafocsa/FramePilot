"""Native Apple Vision face detection, OpenCV contours, and conservative tracking."""

import os
import sys

import cv2
import numpy as np

from .geometry import Rect, iou
from .slot_detection import Slot as Slot
from .slot_detection import detect_slots as detect_slots


class FaceDetector:
    def __init__(self) -> None:
        self.neural = None
        self.native = sys.platform == "darwin" and os.environ.get("FRAMEPILOT_VISION_BACKEND") != "yunet"
        self.backend = "Apple Vision"
        if self.native:
            import Quartz
            import Vision

            self.quartz, self.vision = Quartz, Vision
        else:
            self.backend = "YuNet · OpenCV DNN"

    def neural_detector(self):
        if self.neural is None:
            from .neural_faces import NeuralFaces

            self.neural = NeuralFaces()
        return self.neural

    def detect(self, bgr: np.ndarray) -> list[Rect]:
        h, w = bgr.shape[:2]
        if not self.native:
            return self.neural_detector().detect(bgr)
        ratio = min(1.0, 960 / max(w, h))
        small = cv2.resize(bgr, (max(1, round(w * ratio)), max(1, round(h * ratio))))
        q, v = self.quartz, self.vision
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        sh, sw = rgb.shape[:2]
        # CGDataProvider retains the Python bytes via the PyObjC bridge.
        data = rgb.tobytes()
        provider = q.CGDataProviderCreateWithData(None, data, len(data), None)
        image = q.CGImageCreate(
            sw,
            sh,
            8,
            24,
            sw * 3,
            q.CGColorSpaceCreateDeviceRGB(),
            q.kCGImageAlphaNone,
            provider,
            None,
            False,
            q.kCGRenderingIntentDefault,
        )
        request = v.VNDetectFaceRectanglesRequest.alloc().init()
        handler = v.VNImageRequestHandler.alloc().initWithCGImage_orientation_options_(image, 1, None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Apple Vision: {error}")
        boxes = []
        for observation in request.results() or []:
            b = observation.boundingBox()
            boxes.append(
                Rect(b.origin.x, 1 - b.origin.y - b.size.height, b.size.width, b.size.height).clipped()
            )
        if not boxes:
            boxes = self.neural_detector().detect(bgr)
            self.backend = "Apple Vision + YuNet" if boxes else "Apple Vision"
        else:
            self.backend = "Apple Vision"
        return boxes

    def eyes(self, bgr: np.ndarray, face: Rect) -> tuple[float, float] | None:
        if not self.native:
            return self.neural_detector().eyes(bgr, face)
        q, v = self.quartz, self.vision
        h, w = bgr.shape[:2]
        scale = min(1, 640 / max(w, h))
        rgb = cv2.cvtColor(
            cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale)))), cv2.COLOR_BGR2RGB
        )
        sh, sw = rgb.shape[:2]
        data = rgb.tobytes()
        provider = q.CGDataProviderCreateWithData(None, data, len(data), None)
        image = q.CGImageCreate(
            sw,
            sh,
            8,
            24,
            sw * 3,
            q.CGColorSpaceCreateDeviceRGB(),
            q.kCGImageAlphaNone,
            provider,
            None,
            False,
            q.kCGRenderingIntentDefault,
        )
        observation = v.VNFaceObservation.observationWithBoundingBox_(
            ((face.x, 1 - face.y - face.h), (face.w, face.h))
        )
        request = v.VNDetectFaceLandmarksRequest.alloc().init()
        request.setInputFaceObservations_([observation])
        handler = v.VNImageRequestHandler.alloc().initWithCGImage_orientation_options_(image, 1, None)
        ok, _ = handler.performRequests_error_([request], None)
        if not ok:
            return None
        result = next(iter(request.results() or []), None)
        if result is None or result.landmarks() is None:
            return None
        points = []
        for region in (result.landmarks().leftEye(), result.landmarks().rightEye()):
            if region is not None and region.pointCount():
                values = region.normalizedPoints()
                for index in range(region.pointCount()):
                    point = values[index]
                    points.append((face.x + point.x * face.w, face.y + (1 - point.y) * face.h))
        if not points:
            return None
        return float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))

    def people(self, bgr, upper_body=False):
        if not self.native:
            raise ValueError(
                "Поиск фигуры человека пока доступен в macOS; выберите лицо или выделите предмет"
            )
        from .foreground import vision_image

        v = self.vision
        image, backing = vision_image(bgr)
        request = v.VNDetectHumanRectanglesRequest.alloc().init()
        request.setUpperBodyOnly_(upper_body)
        handler = v.VNImageRequestHandler.alloc().initWithCGImage_orientation_options_(image, 1, None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Поиск человека: {error}")
        rectangles = []
        for item in request.results() or []:
            box = item.boundingBox()
            rectangles.append(
                Rect(
                    box.origin.x, 1 - box.origin.y - box.size.height, box.size.width, box.size.height
                ).clipped()
            )
        return rectangles


class PeopleDetector:
    def __init__(self, detector, upper=False):
        self.detector, self.upper = detector, upper

    def detect(self, image):
        return self.detector.people(image, self.upper)


def appearance(image: np.ndarray, rect: Rect) -> np.ndarray | None:
    h, w = image.shape[:2]
    r = rect.clipped().pixels(w, h)
    crop = image[int(r.y) : int(r.y + r.h), int(r.x) : int(r.x + r.w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    return cv2.normalize(hist, None).flatten()


class FaceTracker:
    """Spatial and appearance gates reduce switching; this is not biometric identification."""

    def __init__(self, detector: FaceDetector) -> None:
        self.detector = detector
        self.reset()

    def reset(self) -> None:
        self.box: Rect | None = None
        self.last_seen = -1e9
        self.missing_since = None
        self.selected = False
        self.reference: np.ndarray | None = None
        self.gray: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.count = 0
        self.proposals = []

    def update(
        self, image: np.ndarray, now: float, click: tuple[float, float] | None = None
    ) -> tuple[Rect | None, list[Rect]]:
        self.proposals = []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        estimate = self.box
        flow_ok = False
        if click is None and self.gray is not None and self.points is not None and len(self.points) >= 6:
            new, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None)
            if new is not None and status is not None:
                good_old = self.points[status.ravel() == 1]
                good_new = new[status.ravel() == 1]
                if len(good_new) >= 6 and self.box:
                    matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                    if matrix is not None and inliers is not None and float(inliers.mean()) > 0.6:
                        r = self.box.pixels(w, h)
                        corners = np.array([[[r.x, r.y], [r.x + r.w, r.y + r.h]]], dtype=np.float32)
                        mapped = cv2.transform(corners, matrix)[0]
                        candidate = Rect(
                            mapped[0, 0] / w,
                            mapped[0, 1] / h,
                            (mapped[1, 0] - mapped[0, 0]) / w,
                            (mapped[1, 1] - mapped[0, 1]) / h,
                        ).clipped()
                        if candidate.area > 0 and iou(candidate, self.box) > 0.3:
                            estimate = candidate
                            flow_ok = True
        faces: list[Rect] = []
        confirmed = False
        must_detect = self.count % 3 == 0 or not flow_ok or click is not None
        if must_detect:
            faces = self.detector.detect(image)
            local_search = (click is not None and not any(r.contains(click) for r in faces)) or (
                click is None
                and self.selected
                and estimate is not None
                and not any(iou(r, estimate) > 0.25 for r in faces)
            )
            if local_search and hasattr(self.detector, "native"):
                # A local retry enlarges a small selected face without changing its identity gate.
                region = (
                    Rect(click[0] - 0.18, click[1] - 0.18, 0.36, 0.36)
                    if click is not None
                    else Rect(
                        estimate.x - estimate.w, estimate.y - estimate.h, estimate.w * 3, estimate.h * 3
                    )
                ).clipped()
                area = region.pixels(w, h)
                x, y, right, bottom = (
                    max(0, round(area.x)),
                    max(0, round(area.y)),
                    min(w, round(area.x + area.w)),
                    min(h, round(area.y + area.h)),
                )
                if right - x >= 16 and bottom - y >= 16:
                    local = self.detector.detect(image[y:bottom, x:right])
                    retry = [
                        Rect(
                            (x + r.x * (right - x)) / w,
                            (y + r.y * (bottom - y)) / h,
                            r.w * (right - x) / w,
                            r.h * (bottom - y) / h,
                        )
                        for r in local
                    ]
                    faces += [r for r in retry if not any(iou(r, existing) > 0.7 for existing in faces)]
            chosen = None
            if click is not None:
                choices = [r for r in faces if r.contains(click)]
                chosen = min(choices, key=lambda r: r.area) if choices else None
                # Clicking empty space leaves tracking unchanged, never chooses another face.
            elif not self.selected:
                chosen = max(faces, key=lambda r: r.area) if faces else None
            elif estimate and (self.missing_since is None or now - self.missing_since < 1.2):
                choices = []
                for face in faces:
                    hist = appearance(image, face)
                    similar = self.reference is None or (
                        hist is not None
                        and cv2.compareHist(self.reference, hist, cv2.HISTCMP_BHATTACHARYYA) < 0.6
                    )
                    if iou(face, estimate) > 0.25 and similar:
                        choices.append(face)
                chosen = max(choices, key=lambda r: iou(r, estimate)) if choices else None
            if chosen:
                estimate = chosen
                if not self.selected or click is not None:
                    self.reference = appearance(image, chosen)
                self.selected = True
                self.last_seen = now
                self.missing_since = None
                confirmed = True
            elif click is None:
                flow_ok = False
        self.count += 1
        visible = estimate if (confirmed or (flow_ok and now - self.last_seen < 0.3)) else None
        if visible:
            self.box = visible
            mask = np.zeros_like(gray)
            r = visible.pixels(w, h)
            mask[max(0, int(r.y)) : min(h, int(r.y + r.h)), max(0, int(r.x)) : min(w, int(r.x + r.w))] = 255
            self.points = cv2.goodFeaturesToTrack(
                gray, maxCorners=60, qualityLevel=0.02, minDistance=5, mask=mask
            )
        else:
            self.points = None
            if self.selected and self.missing_since is None:
                self.missing_since = now
            if self.selected and self.reference is not None and faces:
                ranked = []
                for candidate in faces:
                    histogram = appearance(image, candidate)
                    if histogram is None:
                        continue
                    difference = cv2.compareHist(self.reference, histogram, cv2.HISTCMP_BHATTACHARYYA)
                    ratio = candidate.area / max(self.box.area if self.box else candidate.area, 1e-8)
                    if difference < 0.55 and 0.25 < ratio < 4:
                        ranked.append((difference, candidate))
                self.proposals = [rect for _, rect in sorted(ranked, key=lambda item: item[0])[:3]]
        self.gray = gray
        return visible, faces
