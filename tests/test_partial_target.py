from dataclasses import replace

import cv2
import numpy as np
import pytest

from framepilot.calibration import Calibration, received_preview
from framepilot.geometry import Rect, fit, iou
from framepilot.objects import ObjectTracker
from framepilot.pipeline import Pipeline, Renderer, SceneSlot, Settings


def two_color_object(x=40, y=30):
    image = np.zeros((180, 320, 3), np.uint8)
    image[y : y + 80, x : x + 60] = (0, 0, 240)
    image[y : y + 80, x + 60 : x + 120] = (240, 0, 0)
    return image, Rect(x / 320, y / 180, 120 / 320, 80 / 180)


@pytest.mark.parametrize("fill", [False, True])
def test_half_target_excludes_other_half_even_with_different_destination_aspect(fill):
    image, subject = two_color_object()
    settings = Settings(
        width=400, height=200, subject_mode="object", padding=0, fill=fill, subject_region=Rect(0, 0, 0.5, 1)
    )
    output = Renderer().render(image, subject, settings, 0)
    assert output[:, :, 2].max() == 240
    assert output[:, :, 0].max() == 0


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_half_target_orientation_compensation_keeps_the_original_selected_half(rotation, mirror):
    image, subject = two_color_object()
    calibration = Calibration(crop=Rect(0.1, 0.1, 0.8, 0.8), mirror=mirror, rotation=rotation)
    settings = Settings(
        width=400, height=200, padding=0, subject_region=Rect(0, 0, 0.5, 1), calibration=calibration
    )
    output = received_preview(Renderer().render(image, subject, settings, 0), calibration)
    assert output[:, :, 2].max() == 240
    assert output[:, :, 0].max() == 0


def test_two_slots_can_show_different_parts_of_one_tracked_object():
    image, subject = two_color_object()
    slots = (
        SceneSlot(
            "left", Rect(0, 0, 0.5, 1), subject_mode="object", padding=0, subject_region=Rect(0, 0, 0.5, 1)
        ),
        SceneSlot(
            "right",
            Rect(0.5, 0, 0.5, 1),
            subject_mode="object",
            padding=0,
            subject_region=Rect(0.5, 0, 0.5, 1),
        ),
    )
    pipeline = Pipeline()
    output, _, _ = pipeline.process(
        image, 0, Settings(width=400, height=200, slots=slots), {"left": subject, "right": subject}, True
    )
    assert output[:, :200, 0].max() == 0 and output[:, :200, 2].max() > 230
    assert output[:, 200:, 2].max() == 0 and output[:, 200:, 0].max() > 230
    assert pipeline.last_tracks["left"]["subject"] == subject
    assert pipeline.last_tracks["left"]["tracking"]["state"] == "selected"
    assert pipeline.last_tracks["right"]["tracking"]["state"] == "selected"


def test_responsive_mode_settles_after_movement_without_resetting_tracker():
    first, subject = two_color_object()
    moved, next_subject = two_color_object(x=120)
    settings = Settings(
        width=400, height=300, padding=0, subject_region=Rect(0, 0, 0.5, 1), motion_mode="responsive"
    )
    fast, slow = Renderer(), Renderer()
    fast.render(first, subject, settings, 0)
    slow.render(first, subject, replace(settings, motion_mode="smooth"), 0)
    goal = fit(next_subject.region(settings.subject_region).pixels(320, 180), Rect(0, 0, 400, 300), padding=0)
    fast.render(moved, next_subject, settings, 1 / 30)
    slow.render(moved, next_subject, replace(settings, motion_mode="smooth"), 1 / 30)
    assert abs(fast.placement.x - goal.x) < abs(slow.placement.x - goal.x) / 4
    for index in (2, 3):
        fast.render(moved, next_subject, settings, index / 30)
    assert abs(fast.placement.x - goal.x) < 1


def test_flow_tracks_whole_object_through_motion_scale_and_loss_without_drift():
    texture = np.random.default_rng(14).integers(40, 240, (150, 180, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    tracker = ObjectTracker()
    for index in range(45):
        factor = 1 + 0.12 * np.sin(index / 9)
        width, height = round(180 * factor), round(150 * factor)
        x, y = 40 + index * 3, 80 + index
        image = np.zeros((360, 640, 3), np.uint8)
        image[y : y + height, x : x + width] = cv2.resize(texture, (width, height))
        expected = Rect(x / 640, y / 360, width / 640, height / 360)
        found, _ = tracker.update(image, index / 30, expected if index == 0 else None)
        assert found is not None, index
        assert iou(found, expected) > 0.85, (index, found, expected)
    assert tracker.flow_updates > 30
    assert tracker.update(np.zeros_like(image), 2)[0] is None
    # A different appearance at the former coordinates must not inherit the lock.
    image[y : y + height, x : x + width] = (255, 40, 0)
    assert tracker.update(image, 2.1)[0] is None
