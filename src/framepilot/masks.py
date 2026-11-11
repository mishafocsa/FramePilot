"""Destination masks shared by preview, live output and offline export."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMask:
    kind: str = "auto"
    radius: float = 0.15
    image: str = ""

    def record(self):
        return {"kind": self.kind, "radius": self.radius, "image": self.image}

    @classmethod
    def from_record(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("Неверная маска рамки")
        kind, radius, image = value.get("kind", "auto"), value.get("radius", 0.15), value.get("image", "")
        if kind not in ("auto", "rectangle", "ellipse", "circle", "rounded", "png"):
            raise ValueError("Неизвестная форма маски")
        if type(radius) not in (int, float) or not math.isfinite(radius) or not 0 <= radius <= 0.5:
            raise ValueError("Неверное скругление маски")
        if not isinstance(image, str) or len(image) > 4096 or (kind == "png" and not image):
            raise ValueError("Выберите PNG с прозрачностью")
        return cls(kind, float(radius), image)


def read_alpha(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Маска должна быть PNG с альфа-каналом")
    if max(image.shape[:2]) > 8192 or not np.any(image[:, :, 3]):
        raise ValueError("Пустая или слишком большая маска (максимум 8192 px)")
    return image[:, :, 3].astype(np.float32) / 255


class MaskRenderer:
    def __init__(self):
        self.key = None
        self.value = None

    def alpha(self, size, destination, shape, configuration):
        stamp = Path(configuration.image).stat().st_mtime_ns if configuration.kind == "png" else None
        key = size, destination, shape, configuration, stamp
        if key == self.key:
            return self.value
        width, height = size
        x, y = max(0, round(destination.x)), max(0, round(destination.y))
        right, bottom = (
            min(width, round(destination.x + destination.w)),
            min(height, round(destination.y + destination.h)),
        )
        w, h = right - x, bottom - y
        result = np.zeros((height, width), np.float32)
        if min(w, h) <= 0:
            return result
        kind = shape if configuration.kind == "auto" else configuration.kind
        local = np.zeros((h, w), np.float32)
        if kind == "png":
            local = cv2.resize(read_alpha(configuration.image), (w, h), interpolation=cv2.INTER_LINEAR)
        elif kind in ("ellipse", "circle"):
            axes = (max(1, (w - 1) // 2), max(1, (h - 1) // 2))
            if kind == "circle":
                axes = (min(axes), min(axes))
            cv2.ellipse(local, (w // 2, h // 2), axes, 0, 0, 360, 1, -1)
        elif kind == "rounded":
            radius = min(round(min(w, h) * configuration.radius), min(w, h) // 2)
            local[:] = 1
            if radius:
                for cx, cy, left, top in (
                    (radius, radius, 0, 0),
                    (w - radius - 1, radius, w - radius, 0),
                    (radius, h - radius - 1, 0, h - radius),
                    (w - radius - 1, h - radius - 1, w - radius, h - radius),
                ):
                    local[top : top + radius, left : left + radius] = 0
                    cv2.circle(local, (cx, cy), radius, 1, -1)
        else:
            local[:] = 1
        result[y:bottom, x:right] = local
        self.key, self.value = key, result
        return result
