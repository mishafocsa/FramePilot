from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import framepilot
from framepilot.foreground import Background, PersonSegmenter
from framepilot.geometry import Rect, iou
from framepilot.objects import ObjectTracker
from framepilot.pipeline import Pipeline, Renderer, SceneSlot, Settings
from framepilot.vision import FaceDetector


def test_native_person_mask_and_body_detection():
    photo = cv2.imread(str(Path(framepilot.__file__).parent / "assets/astronaut.png"))
    mask = PersonSegmenter().mask(photo)
    assert mask.shape == photo.shape[:2]
    assert mask.dtype == np.float32
    assert mask.max() > 0.8 and mask.min() < 0.2
    assert 0.05 < mask.mean() < 0.95
    detector = FaceDetector()
    assert detector.people(photo, upper_body=True)
    assert isinstance(detector.people(photo), list)


def test_foreground_alpha_reveals_background_without_black_rectangles():
    image = np.full((100, 100, 3), (0, 0, 240), np.uint8)
    alpha = np.zeros((100, 100), np.float32)
    alpha[30:70, 30:70] = 1
    settings = Settings(width=200, height=200, background_color="#00ff00", padding=0)
    output = Renderer().render(image, Rect(0, 0, 1, 1), settings, 0, foreground=alpha)
    np.testing.assert_array_equal(output[10, 10], (0, 255, 0))
    np.testing.assert_array_equal(output[100, 100], (0, 0, 240))


def test_two_slots_track_separate_objects_and_compose_independently():
    image = np.zeros((180, 300, 3), np.uint8)
    rng = np.random.default_rng(42)
    image[50:100, 30:80] = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
    image[50:100, 200:250] = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
    slots = (
        SceneSlot("left", Rect(0, 0, 0.45, 1), subject_mode="object"),
        SceneSlot("right", Rect(0.55, 0, 0.45, 1), subject_mode="object"),
    )
    settings = Settings(
        width=400, height=240, slots=slots, active_slot="left", mapping="canvas", background_color="#00ff00"
    )
    pipeline = Pipeline()
    output, first, _ = pipeline.process(
        image,
        0,
        settings,
        {
            "left": Rect(30 / 300, 50 / 180, 50 / 300, 50 / 180),
            "right": Rect(200 / 300, 50 / 180, 50 / 300, 50 / 180),
        },
        True,
    )
    assert first is not None
    assert pipeline.last_tracks["right"]["subject"] is not None
    assert np.any(output[:, :160] != (0, 255, 0))
    assert np.any(output[:, 240:] != (0, 255, 0))
    np.testing.assert_array_equal(output[120, 200], (0, 255, 0))
    right_before = pipeline.children["right"].object_tracker.box
    pipeline.process(image, 0.1, settings, {"_reset": "left"}, False, repaint=True)
    assert pipeline.children["right"].object_tracker.box == right_before
    assert pipeline.children["left"].object_tracker.box is None


def test_freeze_keeps_subject_but_allows_background_color_change():
    renderer = Renderer()
    image = np.full((100, 100, 3), 200, np.uint8)
    settings = Settings(width=200, height=200, mapping="canvas", target=Rect(0.25, 0.25, 0.5, 0.5))
    renderer.render(image, Rect(0.2, 0.2, 0.6, 0.6), settings, 0)
    output = renderer.render(image, None, replace(settings, background_color="#ff0000"), 0.1)
    np.testing.assert_array_equal(output[0, 0], (0, 0, 255))
    assert output[100, 100].mean() > 100


def test_transparent_background_png_composites_over_selected_color(tmp_path):
    image = np.zeros((50, 100, 4), np.uint8)
    image[:, 50:] = (255, 0, 0, 255)
    path = tmp_path / "template.png"
    assert cv2.imwrite(str(path), image)
    background = Background()
    settings = Settings(width=200, height=100, background_color="#00ff00", background_image=str(path))
    output = background.frame(settings)
    np.testing.assert_array_equal(output[40, 20], (0, 255, 0))
    np.testing.assert_array_equal(output[40, 180], (255, 0, 0))
    changed = background.frame(replace(settings, background_color="#ff0000"))
    np.testing.assert_array_equal(changed[40, 20], (0, 0, 255))


def test_long_object_loss_proposes_distant_match_but_requires_confirmation():
    rng = np.random.default_rng(13)
    patch = rng.integers(40, 255, (40, 40, 3), dtype=np.uint8)
    original = np.zeros((160, 400, 3), np.uint8)
    original[50:90, 20:60] = patch
    moved = np.zeros_like(original)
    moved[50:90, 320:360] = patch
    tracker = ObjectTracker()
    assert tracker.update(original, 0, Rect(0.05, 0.3125, 0.1, 0.25))[0] is not None
    assert tracker.update(np.zeros_like(original), 0.1)[0] is None
    assert tracker.update(moved, 2)[0] is None
    assert tracker.proposals
    assert iou(tracker.proposals[0], Rect(0.8, 0.3125, 0.1, 0.25)) > 0.9
    assert tracker.update(moved, 2.1, tracker.proposals[0])[0] is not None


def test_original_plan_on_loss_fits_its_own_slot():
    image = np.full((100, 100, 3), (0, 0, 240), np.uint8)
    settings = Settings(
        width=400, height=200, mapping="canvas", target=Rect(0, 0, 0.25, 0.5), loss="original"
    )
    output = Renderer().render(image, None, settings, 0)
    np.testing.assert_array_equal(output[50, 50], (0, 0, 240))
    assert not output[150, 150].any()
