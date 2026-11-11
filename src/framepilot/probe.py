"""Session-specific video test pattern, observed return timing, and crop/orientation measurement."""

import binascii
import random
import secrets
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .calibration import Calibration
from .geometry import Rect


@dataclass(frozen=True)
class ProbeObservation:
    sequence: int | None
    observed_delay_ms: float | None
    capture_ms: float
    markers: int
    crop: Rect
    mirror: bool
    rotation: int
    reprojection_px: float
    image_width: int
    image_height: int

    def record(self):
        value = asdict(self)
        value["crop"] = [self.crop.x, self.crop.y, self.crop.w, self.crop.h]
        return value

    def calibration(self):
        return Calibration(self.crop, self.mirror, rotation=self.rotation)


class VideoProbe:
    def __init__(self, width=1280, height=720, token=None):
        if width < 160 or height < 160:
            raise ValueError("Для проверочной сетки требуется не менее 160×160")
        self.width, self.height = width, height
        self.token = secrets.randbits(32) if token is None else token
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)
        self.marker_points = {}
        self.base = np.full((height, width, 3), 235, np.uint8)
        edge = max(28, round(min(width, height) * 0.13))
        ids = random.Random(self.token).sample(range(1000), 8)
        positions = [
            (0.12, 0.12),
            (0.5, 0.12),
            (0.88, 0.12),
            (0.12, 0.5),
            (0.88, 0.5),
            (0.12, 0.88),
            (0.5, 0.88),
            (0.88, 0.88),
        ]
        for identifier, (cx, cy) in zip(ids, positions, strict=True):
            x, y = round(cx * width - edge / 2), round(cy * height - edge / 2)
            marker = cv2.aruco.generateImageMarker(self.dictionary, identifier, edge)
            self.base[y : y + edge, x : x + edge] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            self.marker_points[identifier] = np.float32(
                [[x, y], [x + edge - 1, y], [x + edge - 1, y + edge - 1], [x, y + edge - 1]]
            )
        self.cell = max(4, round(min(width, height) * 0.028))
        self.stamp_x = (width - self.cell * 12) // 2
        self.stamp_y = (height - self.cell * 10) // 2
        self.sequence = 0
        self.emitted = OrderedDict()
        self.lock = threading.Lock()

    def frame(self, now=None):
        now = time.monotonic() if now is None else now
        sequence = self.sequence
        self.sequence += 1
        payload = struct.pack(">II", self.token, sequence)
        bits = np.unpackbits(
            np.frombuffer(payload + struct.pack(">H", binascii.crc_hqx(payload, 0xFFFF)), np.uint8)
        )
        stamp = np.full((10, 12), 255, np.uint8)
        stamp[1:9, 1:11] = bits.reshape(8, 10) * 255
        stamp = cv2.resize(stamp, (self.cell * 12, self.cell * 10), interpolation=cv2.INTER_NEAREST)
        image = self.base.copy()
        x, y = self.stamp_x, self.stamp_y
        image[y : y + stamp.shape[0], x : x + stamp.shape[1]] = cv2.cvtColor(stamp, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            image,
            f"FramePilot   {sequence:06d}",
            (max(5, self.width // 3), self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, min(self.width, self.height) / 1200),
            (80, 35, 15),
            1,
        )
        with self.lock:
            self.emitted[sequence] = now
            while len(self.emitted) > 900:
                self.emitted.popitem(last=False)
        return image, sequence

    def _matches(self, gray):
        best = None
        for flipped in (False, True):
            view = cv2.flip(gray, 1) if flipped else gray
            corners, identifiers, _ = self.detector.detectMarkers(view)
            source, destination, matched = [], [], 0
            used = set()
            for points, identifier in zip(
                corners, identifiers.ravel() if identifiers is not None else [], strict=False
            ):
                identifier = int(identifier)
                if identifier not in self.marker_points or identifier in used:
                    continue
                used.add(identifier)
                target = points.reshape(4, 2).copy()
                if flipped:
                    target[:, 0] = gray.shape[1] - 1 - target[:, 0]
                source.extend(self.marker_points[identifier])
                destination.extend(target)
                matched += 1
            if matched >= 3 and (best is None or matched > best[2]):
                best = np.float32(source), np.float32(destination), matched
        return best

    def observe(self, image, visible: Rect | None = None, observed_at=None, capture_started=None):
        observed_at = time.monotonic() if observed_at is None else observed_at
        capture_started = observed_at if capture_started is None else capture_started
        h, w = image.shape[:2]
        roi = (visible or Rect(0, 0, 1, 1)).clipped().pixels(w, h)
        x, y = max(0, round(roi.x)), max(0, round(roi.y))
        x2, y2 = min(w, round(roi.x + roi.w)), min(h, round(roi.y + roi.h))
        if min(x2 - x, y2 - y) < 80:
            raise ValueError("Выделите видимую область камеры размером не менее 80×80")
        region = image[y:y2, x:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        matches = self._matches(gray)
        if matches is None:
            raise ValueError(
                "Сетка этой проверки не найдена: покажите камеру и оставьте видимыми хотя бы три метки"
            )
        source, destination, marker_count = matches
        matrix, inliers = cv2.estimateAffine2D(
            source, destination, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=1500
        )
        if matrix is None or inliers is None or float(inliers.mean()) < 0.8:
            raise ValueError("Метки не образуют согласованную плоскость; проверьте выделенную область")
        prediction = cv2.transform(source[None], matrix)[0]
        errors = np.linalg.norm(prediction - destination, axis=1)[inliers.ravel() != 0]
        reprojection = float(np.percentile(errors, 90))
        if reprojection > 2.5:
            raise ValueError("Изображение слишком искажено для точной калибровки")
        linear = matrix[:, :2]
        sx, sy = np.linalg.norm(linear[:, 0]), np.linalg.norm(linear[:, 1])
        if (
            min(sx, sy) < 0.01
            or abs(sx / sy - 1) > 0.04
            or abs(np.dot(linear[:, 0], linear[:, 1]) / (sx * sy)) > 0.04
        ):
            raise ValueError("Приёмник растягивает или наклоняет изображение: согласуйте пропорции вручную")
        mirror = bool(np.linalg.det(linear) < 0)
        angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))) - (180 if mirror else 0)
        rotation = (round(angle / 90) * 90) % 360
        difference = abs((angle - rotation + 180) % 360 - 180)
        if difference > 2:
            raise ValueError("Поддерживается поворот на 0°, 90°, 180° или 270°")
        inverse = cv2.invertAffineTransform(matrix)
        visible_points = cv2.transform(
            np.float32(
                [[[0, 0], [region.shape[1], 0], [region.shape[1], region.shape[0]], [0, region.shape[0]]]]
            ),
            inverse,
        )[0]
        low = np.maximum(visible_points.min(axis=0) / [self.width, self.height], 0)
        high = np.minimum(visible_points.max(axis=0) / [self.width, self.height], 1)
        crop = Rect(float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1]))
        if min(crop.w, crop.h) < 0.05:
            raise ValueError("Слишком малая видимая часть камеры")
        points = np.float32(
            [
                [
                    [self.stamp_x + (column + 1.5) * self.cell, self.stamp_y + (row + 1.5) * self.cell]
                    for row in range(8)
                    for column in range(10)
                ]
            ]
        )
        samples = cv2.transform(points, matrix)[0]
        sequence, delay = None, None
        if (
            samples[:, 0].min() >= 1
            and samples[:, 1].min() >= 1
            and samples[:, 0].max() < gray.shape[1] - 1
            and samples[:, 1].max() < gray.shape[0] - 1
        ):
            values = cv2.remap(
                gray,
                samples[:, 0].reshape(8, 10),
                samples[:, 1].reshape(8, 10),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            packed = np.packbits((values.ravel() > 127).astype(np.uint8)).tobytes()
            token, candidate, checksum = struct.unpack(">IIH", packed)
            if token == self.token and checksum == binascii.crc_hqx(packed[:8], 0xFFFF):
                with self.lock:
                    emitted = self.emitted.get(candidate)
                if emitted is not None and observed_at >= emitted:
                    sequence = candidate
                    delay = (observed_at - emitted) * 1000
        return ProbeObservation(
            sequence,
            delay,
            max(0, observed_at - capture_started) * 1000,
            marker_count,
            crop,
            mirror,
            rotation,
            reprojection,
            w,
            h,
        )


class ProbeHistory:
    def __init__(self):
        self.rows = []
        self.geometry = []
        self.seen = set()
        self.last_new_at = None
        self.repeated = 0

    def add(self, observation, now=None):
        now = time.monotonic() if now is None else now
        self.geometry.append(observation)
        self.geometry = self.geometry[-30:]
        if observation.sequence is None:
            return False
        if observation.sequence in self.seen:
            self.repeated += 1
            return False
        self.seen.add(observation.sequence)
        self.last_new_at = now
        self.rows.append(observation)
        if len(self.rows) > 900:
            removed = self.rows.pop(0)
            self.seen.discard(removed.sequence)
        return True

    def stable_calibration(self):
        rows = self.geometry[-3:]
        if len(rows) < 3 or len({(r.mirror, r.rotation) for r in rows}) != 1:
            return None
        values = np.array([[r.crop.x, r.crop.y, r.crop.w, r.crop.h] for r in rows])
        if np.max(np.ptp(values, axis=0)) > 0.01:
            return None
        crop = np.mean(values, axis=0)
        return Calibration(
            Rect(*(float(value) for value in crop)), rows[-1].mirror, rotation=rows[-1].rotation
        )

    def summary(self):
        timings = [row.observed_delay_ms for row in self.rows if row.observed_delay_ms is not None]
        return {
            "matched_unique_frames": len(timings),
            "repeated_observations": self.repeated,
            "median_observed_delay_ms": round(float(np.median(timings)), 2) if timings else None,
            "p95_observed_delay_ms": round(float(np.percentile(timings, 95)), 2) if timings else None,
            "mean_capture_ms": round(float(np.mean([r.capture_ms for r in self.rows])), 2)
            if self.rows
            else None,
            "scope": "Output pattern emission to observed return, including return sampling/capture; not sensor-to-display latency",
        }
