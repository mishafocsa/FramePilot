"""Bundled YuNet face detector. Runs locally through OpenCV DNN on both platforms."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from .geometry import Rect, iou


@dataclass(frozen=True)
class NeuralFace:
    rect: Rect
    eyes: tuple[float, float]
    score: float


class NeuralFaces:
    def __init__(self):
        directory = Path(__file__).parent / "assets"
        manifest = json.loads((directory / "yunet.json").read_text(encoding="utf-8"))
        model = directory / manifest["file"]
        if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError("Модель лица повреждена; переустановите FramePilot")
        self.detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.8, 0.3, 5000)

    def observations(self, image, max_side=960):
        h, w = image.shape[:2]
        if min(h, w) < 16:
            return []
        scale = min(1, max_side / max(h, w))
        small = cv2.resize(image, (max(16, round(w * scale)), max(16, round(h * scale))))
        sh, sw = small.shape[:2]
        self.detector.setInputSize((sw, sh))
        _, rows = self.detector.detect(small)
        result = []
        for row in rows if rows is not None else ():
            x, y, width, height = (float(v) for v in row[:4])
            rect = Rect(x / sw, y / sh, width / sw, height / sh).clipped()
            if rect.area <= 0:
                continue
            eyes = (float(row[4] + row[6]) / (2 * sw), float(row[5] + row[7]) / (2 * sh))
            result.append(NeuralFace(rect, eyes, float(row[-1])))
        if not result and max_side > 320 and max(h, w) > 320:
            return self.observations(image, 320)
        return result

    def detect(self, image):
        return [face.rect for face in self.observations(image)]

    def eyes(self, image, face):
        found = self.observations(image, 640)
        chosen = max(found, key=lambda item: iou(face, item.rect), default=None)
        return chosen.eyes if chosen is not None and iou(chosen.rect, face) > 0.25 else None
