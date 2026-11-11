"""Deterministic synthetic object video for a reproducible check; not a recognition model."""

from functools import lru_cache

import cv2
import numpy as np

from .geometry import Rect


@lru_cache(maxsize=1)
def object_picture():
    image = np.full((220, 200, 3), (30, 25, 22), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(mask, (100, 126), (84, 86), 0, 0, 360, 255, -1)
    noise = np.random.default_rng(24).normal(0, 17, image.shape[:2])
    for channel, base in enumerate((38, 55, 205)):
        image[:, :, channel][mask != 0] = np.clip(base + noise[mask != 0], 0, 255).astype(np.uint8)
    for x, y in np.random.default_rng(8).integers((25, 60), (175, 195), (90, 2)):
        if mask[y, x]:
            cv2.circle(image, (int(x), int(y)), 2, (55, 95, 235), -1)
    cv2.ellipse(image, (62, 98), (12, 27), 22, 0, 360, (110, 160, 250), -1)
    cv2.line(image, (100, 51), (109, 15), (45, 85, 105), 8)
    cv2.ellipse(image, (133, 29), (31, 12), -20, 0, 360, (55, 155, 65), -1)
    return image


def object_frame(index):
    picture = object_picture()
    factor = 0.9 + 0.1 * np.sin(index / 23)
    width, height = round(picture.shape[1] * factor), round(picture.shape[0] * factor)
    x = 90 + round(80 * np.sin(index / 25))
    y = 65 + round(22 * np.sin(index / 18))
    frame = np.full((360, 640, 3), (30, 25, 22), np.uint8)
    frame[y : y + height, x : x + width] = cv2.resize(picture, (width, height))
    return frame, Rect(x / 640, y / 360, width / 640, height / 360)
