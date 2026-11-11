"""Model of the receiving application's crop and horizontal mirroring."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Rect


@dataclass(frozen=True)
class Calibration:
    crop: Rect = Rect(0, 0, 1, 1)
    mirror: bool = False
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: int = 0  # Clockwise rotation applied by the receiver, after crop and mirror.


def rotate_image(image, rotation):
    kind = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(image, kind[rotation % 360]) if rotation % 360 else image


def receiver_to_output(point, calibration):
    x, y = point
    x, y = {0: (x, y), 90: (y, 1 - x), 180: (1 - x, 1 - y), 270: (1 - y, x)}[calibration.rotation]
    return (1 - x, y) if calibration.mirror else (x, y)


def compensate_rect(rect, calibration):
    points = [
        receiver_to_output(point, calibration)
        for point in [
            (rect.x, rect.y),
            (rect.x + rect.w, rect.y),
            (rect.x + rect.w, rect.y + rect.h),
            (rect.x, rect.y + rect.h),
        ]
    ]
    x, y = min(p[0] for p in points), min(p[1] for p in points)
    return Rect(x, y, max(p[0] for p in points) - x, max(p[1] for p in points) - y)


def received_preview(image: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Simulated receiver, not an observation or guarantee about an external app."""
    h, w = image.shape[:2]
    r = calibration.crop.clipped().pixels(w, h)
    x, y = min(w - 1, max(0, round(r.x))), min(h - 1, max(0, round(r.y)))
    x2, y2 = min(w, max(x + 1, round(r.x + r.w))), min(h, max(y + 1, round(r.y + r.h)))
    crop = image[y:y2, x:x2].copy()
    return rotate_image(cv2.flip(crop, 1) if calibration.mirror else crop, calibration.rotation)


def calibration_card(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), (25, 35, 50), np.uint8)
    colors = [(80, 190, 255), (130, 240, 120)]
    for n in range(1, 10):
        x, y = round(width * n / 10), round(height * n / 10)
        cv2.line(image, (x, 0), (x, height - 1), (65, 80, 100), 1)
        cv2.line(image, (0, y), (width - 1, y), (65, 80, 100), 1)
    thickness = max(2, width // 250)
    cv2.rectangle(image, (2, 2), (width - 3, height - 3), colors[0], thickness)
    cv2.line(image, (width // 2, 0), (width // 2, height), colors[1], thickness)
    cv2.line(image, (0, height // 2), (width, height // 2), colors[1], thickness)
    font_scale = max(0.4, min(width, height) / 600)
    for label, position in [
        ("LEFT", (12, 40)),
        ("RIGHT", (max(12, width - 140), 40)),
        ("TOP", (width // 2 + 10, 40)),
        ("BOTTOM", (width // 2 + 10, max(45, height - 20))),
    ]:
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[0], 2)
    return image
