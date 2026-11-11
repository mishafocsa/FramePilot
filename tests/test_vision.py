import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import framepilot
from framepilot.geometry import Rect, iou
from framepilot.vision import FaceDetector, FaceTracker, detect_slots


@pytest.mark.parametrize("shape", ["rectangle", "ellipse"])
def test_detects_real_contour_and_ignores_window_boundary(shape):
    image = np.full((500, 800, 3), 35, np.uint8)
    target = Rect(0.3, 0.1, 0.35, 0.75)
    r = target.pixels(800, 500)
    if shape == "rectangle":
        cv2.rectangle(image, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), (255, 255, 255), 4)
    else:
        cv2.ellipse(
            image,
            (int(r.center[0]), int(r.center[1])),
            (int(r.w / 2), int(r.h / 2)),
            0,
            0,
            360,
            (255, 255, 255),
            4,
        )
    slots = detect_slots(image)
    assert slots
    assert slots[0].shape == shape
    assert iou(slots[0].rect, target) > 0.92


def test_empty_image_has_no_frame():
    assert detect_slots(np.zeros((500, 800, 3), np.uint8)) == []


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return self.faces


def test_select_specific_face_then_refuse_different_person():
    small, large = Rect(0.1, 0.1, 0.15, 0.25), Rect(0.6, 0.2, 0.3, 0.5)
    detector = FakeDetector([small, large])
    tracker = FaceTracker(detector)
    image = np.full((400, 600, 3), 100, np.uint8)
    assert tracker.update(image, 0)[0] == large
    assert tracker.update(image, 0.1, small.center)[0] == small
    detector.faces = [large]
    assert tracker.update(image, 0.2)[0] is None
    assert tracker.update(image, 2)[0] is None


def test_brief_occlusion_can_reacquire_same_face():
    face = Rect(0.2, 0.2, 0.3, 0.4)
    detector = FakeDetector([face])
    tracker = FaceTracker(detector)
    image = np.full((200, 300, 3), 120, np.uint8)
    assert tracker.update(image, 0)[0] == face
    detector.faces = []
    assert tracker.update(image, 0.2)[0] is None
    detector.faces = [face]
    assert tracker.update(image, 0.5)[0] == face


def test_long_face_loss_proposals_do_not_silently_switch_identity():
    first, distant = Rect(0.1, 0.1, 0.2, 0.3), Rect(0.7, 0.1, 0.2, 0.3)
    detector = FakeDetector([first])
    tracker = FaceTracker(detector)
    image = np.full((200, 400, 3), 120, np.uint8)
    assert tracker.update(image, 0)[0] == first
    detector.faces = []
    assert tracker.update(image, 0.1)[0] is None
    detector.faces = [distant]
    assert tracker.update(image, 2)[0] is None
    assert tracker.proposals == [distant]
    assert tracker.update(image, 2.1, distant.center)[0] == distant


def test_decoder_skipped_time_is_not_a_confirmed_occlusion():
    face = Rect(0.2, 0.2, 0.3, 0.4)
    detector = FakeDetector([face])
    tracker = FaceTracker(detector)
    image = np.full((200, 300, 3), 120, np.uint8)
    assert tracker.update(image, 0)[0] == face
    # No intermediate frames were observed, e.g. during a cold model start.
    assert tracker.update(image, 2)[0] == face


def test_appearance_rejects_different_subject_at_same_position():
    face = Rect(0.2, 0.2, 0.3, 0.4)
    detector = FakeDetector([face])
    tracker = FaceTracker(detector)
    first = np.zeros((200, 300, 3), np.uint8)
    first[:] = (30, 80, 220)
    assert tracker.update(first, 0)[0] == face
    second = np.zeros_like(first)
    second[:] = (220, 80, 30)
    assert tracker.update(second, 0.1)[0] is None


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision integration")
def test_native_apple_vision_can_process_real_pixel_data():
    detector = FaceDetector()
    assert detector.backend == "Apple Vision"
    assert detector.detect(np.zeros((240, 320, 3), np.uint8)) == []


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision integration")
def test_native_face_tracking_follows_motion_and_recovers_after_short_occlusion():
    image = cv2.imread(str(Path(framepilot.__file__).parent / "assets" / "astronaut.png"))
    assert image is not None
    detector = FaceDetector()
    faces = detector.detect(image)
    assert faces, "Apple Vision must find an actual face, not just accept the image buffer"
    original = max(faces, key=lambda face: face.area)
    tracker = FaceTracker(detector)
    for index in range(18):
        dx = index * 2
        shifted = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (512, 512))
        found, _ = tracker.update(shifted, index / 30)
        expected = Rect(original.x + dx / 512, original.y, original.w, original.h)
        assert found is not None
        assert iou(found, expected) > 0.65
    hidden = np.zeros_like(image)
    assert tracker.update(hidden, 0.65)[0] is None
    restored = cv2.warpAffine(image, np.float32([[1, 0, 34], [0, 1, 0]]), (512, 512))
    assert tracker.update(restored, 0.8)[0] is not None
    assert tracker.update(hidden, 1)[0] is None
    assert tracker.update(restored, 3)[0] is None
