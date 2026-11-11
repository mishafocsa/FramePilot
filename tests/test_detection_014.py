import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

import framepilot
from framepilot.geometry import Rect, iou
from framepilot.neural_faces import NeuralFaces
from framepilot.vision import FaceDetector, FaceTracker, detect_slots


@pytest.mark.parametrize("kind", ["thin", "dashed", "large", "blurred", "toolbar", "small_roi"])
def test_frame_styles_and_toolbar_distractor(kind):
    image = np.full((600, 900, 3), 70, np.uint8)
    x, y, width, height = 216, 78, 450, 438
    roi = None
    if kind == "large":
        x, y, width, height = 18, 12, 864, 576
    elif kind == "small_roi":
        x, y, width, height = 680, 440, 80, 95
        roi = Rect(0.71, 0.69, 0.2, 0.3)
    if kind == "dashed":
        for xx in range(x, x + width, 14):
            for yy in (y, y + height):
                cv2.line(image, (xx, yy), (min(xx + 8, x + width), yy), (115, 115, 115), 1)
        for yy in range(y, y + height, 14):
            for xx in (x, x + width):
                cv2.line(image, (xx, yy), (xx, min(yy + 8, y + height)), (115, 115, 115), 1)
    else:
        cv2.rectangle(image, (x, y), (x + width, y + height), (91, 91, 91), 1)
    if kind == "toolbar":
        cv2.rectangle(image, (20, 20), (880, 47), (255, 255, 255), 2)
    if kind == "blurred":
        image = cv2.GaussianBlur(image, (13, 13), 2)
    expected = Rect(x / 900, y / 600, width / 900, height / 600)
    found = detect_slots(image, roi, "sensitive")
    assert found and iou(found[0].rect, expected) > 0.94
    assert len(found) == 1


def test_toolbar_alone_is_not_a_video_frame_but_explicit_roi_allows_narrow_frame():
    image = np.zeros((600, 900, 3), np.uint8)
    cv2.rectangle(image, (100, 230), (800, 265), (250, 250, 250), 2)
    assert detect_slots(image) == []
    found = detect_slots(image, Rect(0.06, 0.32, 0.88, 0.2))
    assert found and iou(found[0].rect, Rect(100 / 900, 230 / 600, 700 / 900, 35 / 600)) > 0.8


@pytest.fixture(scope="module")
def neural():
    return NeuralFaces()


@pytest.mark.parametrize("brightness,scale", [(1, 1), (0.4, 1), (0.7, 0.5), (1, 0.35)])
def test_bundled_neural_detector_finds_real_face_and_eyes(neural, brightness, scale):
    image = cv2.imread(str(Path(framepilot.__file__).parent / "assets/astronaut.png"))
    image = np.clip(image.astype(float) * brightness, 0, 255).astype(np.uint8)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    found = neural.observations(image)
    expected = Rect(0.34, 0.12, 0.19, 0.23)
    face = max(found, key=lambda item: iou(expected, item.rect))
    assert iou(expected, face.rect) > 0.6 and face.score >= 0.8
    assert face.rect.contains(face.eyes)


def test_portable_detector_never_uses_haar_or_apple(monkeypatch):
    monkeypatch.setenv("FRAMEPILOT_VISION_BACKEND", "yunet")
    detector = FaceDetector()
    assert not detector.native and "YuNet" in detector.backend
    assert detector.detect(np.zeros((180, 320, 3), np.uint8)) == []
    with pytest.raises(ValueError, match="macOS"):
        detector.people(np.zeros((180, 320, 3), np.uint8))


def test_model_is_bundled_with_license_and_pinned_revision():
    root = Path(framepilot.__file__).parent / "assets"
    manifest = json.loads((root / "yunet.json").read_text())
    assert len(manifest["commit"]) == 40
    assert manifest["runtime_downloads"] is False
    assert "MIT License" in (root / "YUNET_LICENSE.txt").read_text()


def test_click_retries_small_region_even_when_other_face_was_found_globally():
    selected = Rect(0.1, 0.2, 0.2, 0.35)

    class Detector:
        native = True

        def detect(self, image):
            if image.shape[:2] == (200, 300):
                return [Rect(0.7, 0.2, 0.2, 0.3)]
            # Click at (0.2, 0.375) creates the 108×72 ROI starting at pixel (6,39).
            return [Rect((30 - 6) / 108, (40 - 39) / 72, 60 / 108, 70 / 72)]

    tracker = FaceTracker(Detector())
    box, candidates = tracker.update(np.full((200, 300, 3), 120, np.uint8), 0, selected.center)
    assert box is not None and iou(box, selected) > 0.99
    assert len(candidates) == 2
