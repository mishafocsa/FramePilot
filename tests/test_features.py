from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import framepilot
from framepilot.calibration import Calibration, received_preview
from framepilot.geometry import Rect, iou
from framepilot.objects import ObjectTracker
from framepilot.pipeline import Renderer, Settings
from framepilot.vision import FaceDetector


def test_crop_and_mirror_compensation_preserves_receiver_orientation():
    image = np.zeros((200, 200, 3), np.uint8)
    image[50:150, 50:100] = (0, 0, 255)
    image[50:150, 100:150] = (255, 0, 0)
    calibration = Calibration(Rect(0.25, 0.1, 0.5, 0.8), mirror=True)
    settings = Settings(width=400, height=400, padding=0, calibration=calibration)
    output = Renderer().render(image, Rect(0.25, 0.25, 0.5, 0.5), settings, 0)
    received = received_preview(output, calibration)
    assert received.shape == (320, 200, 3)
    assert received[160, 50, 2] > 200
    assert received[160, 150, 0] > 200
    assert not output[:, :99].any()


def test_eye_anchor_and_zoom_limit_are_applied():
    image = np.full((300, 400, 3), 100, np.uint8)
    face = Rect(0.3, 0.2, 0.2, 0.4)
    eyes = (0.4, 0.32)
    settings = Settings(width=800, height=600, eye_level=0.3, max_zoom=12)
    renderer = Renderer()
    renderer.render(image, face, settings, 0, eyes=eyes)
    placement = renderer.placement
    assert eyes[1] * 300 * placement.scale + placement.y == pytest.approx(180)
    renderer.render(image, face, replace(settings, max_zoom=1), 0.1, snap=True, eyes=eyes)
    assert renderer.placement.scale <= 2


def test_object_tracking_motion_short_occlusion_and_expiration():
    rng = np.random.default_rng(21)
    patch = rng.integers(60, 255, (50, 60, 3), dtype=np.uint8)
    tracker = ObjectTracker()

    def picture(x):
        image = np.zeros((180, 260, 3), np.uint8)
        image[60:110, x : x + 60] = patch
        return image

    first = Rect(40 / 260, 60 / 180, 60 / 260, 50 / 180)
    assert tracker.update(picture(40), 0, first)[0] is not None
    for index in range(1, 15):
        box, _ = tracker.update(picture(40 + 3 * index), index / 30)
        assert box is not None
        assert iou(box, Rect((40 + 3 * index) / 260, 60 / 180, 60 / 260, 50 / 180)) > 0.8
    blank = np.zeros((180, 260, 3), np.uint8)
    assert tracker.update(blank, 0.6)[0] is None
    assert tracker.update(picture(82), 0.8)[0] is None  # First consistent recovery observation.
    assert tracker.update(picture(82), 0.84)[0] is not None
    assert tracker.update(blank, 1)[0] is None
    assert tracker.update(picture(82), 3)[0] is None


def test_native_eye_landmarks_are_inside_selected_face():
    portrait = cv2.imread(str(Path(framepilot.__file__).parent / "assets/astronaut.png"))
    detector = FaceDetector()
    face = max(detector.detect(portrait), key=lambda r: r.area)
    eyes = detector.eyes(portrait, face)
    assert eyes is not None
    assert face.contains(eyes)
    assert face.y < eyes[1] < face.y + face.h * 0.7
