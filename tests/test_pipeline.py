from dataclasses import replace

import cv2
import numpy as np
import pytest

from framepilot.geometry import Rect
from framepilot.pipeline import Renderer, Settings


def sample():
    image = np.full((200, 300, 3), (30, 60, 90), np.uint8)
    cv2.rectangle(image, (100, 50), (150, 130), (0, 0, 255), -1)
    return image, Rect(100 / 300, 50 / 200, 50 / 300, 80 / 200)


def test_canvas_clips_video_to_destination():
    image, face = sample()
    settings = Settings(width=640, height=360, target=Rect(0.5, 0.2, 0.3, 0.6), mapping="canvas")
    output = Renderer().render(image, face, settings, 0)
    assert output.shape == (360, 640, 3)
    assert not output[:, :300].any()
    assert not output[:, 520:].any()
    assert output[180, 416, 2] > 200


def test_camera_mapping_does_not_apply_window_offset_twice():
    image, face = sample()
    renderer = Renderer()
    settings = Settings(width=640, height=360, target=Rect(0.75, 0.2, 0.2, 0.6), mapping="camera")
    output = renderer.render(image, face, settings, 0)
    assert output[180, 320, 2] > 200


@pytest.mark.parametrize("mode", ["freeze", "black", "original"])
def test_lost_face_behavior(mode):
    image, face = sample()
    renderer = Renderer()
    settings = Settings(width=640, height=360, loss=mode)
    previous = renderer.render(image, face, settings, 0)
    output = renderer.render(image, None, settings, 0.1)
    if mode == "freeze":
        np.testing.assert_array_equal(output, previous)
    elif mode == "black":
        assert not output.any()
    else:
        assert output.any()
        assert not np.array_equal(output, previous)


def test_resolution_change_invalidates_held_frame():
    image, face = sample()
    renderer = Renderer()
    first = Settings(width=640, height=360)
    renderer.render(image, face, first, 0)
    output = renderer.render(image, None, replace(first, width=360, height=640), 0.1)
    assert output.shape == (640, 360, 3)
    assert not output.any()


def test_ellipse_masks_corners():
    image, face = sample()
    output = Renderer().render(image, face, Settings(width=640, height=360, shape="ellipse"), 0)
    assert not output[0, 0].any()
    assert output[180, 320].any()
