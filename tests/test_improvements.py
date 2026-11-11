import cv2
import numpy as np

from framepilot.geometry import Rect, StableTarget, iou
from framepilot.vision import detect_slots


def test_frame_jitter_does_not_resize_output():
    stable = StableTarget()
    box = Rect(0.2, 0.2, 0.4, 0.5)
    stable.update([box])
    stable.update([box])
    jitter = Rect(0.201, 0.199, 0.401, 0.501)
    assert stable.update([jitter]) == box


def test_low_contrast_rounded_frame_is_detected():
    image = np.full((500, 800, 3), 100, np.uint8)
    # Rounded rectangle with only 18 gray levels of contrast.
    cv2.rectangle(image, (250, 70), (530, 430), (118, 118, 118), -1)
    image = cv2.GaussianBlur(image, (25, 25), 5)
    slots = detect_slots(image, sensitivity="sensitive")
    assert slots and slots[0].shape == "rectangle"
    assert iou(slots[0].rect, Rect(250 / 800, 70 / 500, 280 / 800, 360 / 500)) > 0.85


def test_search_region_excludes_larger_unrelated_panel():
    image = np.zeros((500, 800, 3), np.uint8)
    cv2.rectangle(image, (20, 20), (450, 470), (255, 255, 255), 4)
    cv2.rectangle(image, (550, 120), (740, 390), (255, 255, 255), 4)
    slots = detect_slots(image, roi=Rect(0.65, 0.1, 0.34, 0.8))
    assert slots
    assert all(slot.rect.x > 0.65 for slot in slots)
