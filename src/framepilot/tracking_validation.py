"""Deterministic 2-D fixtures and ground-truth checks, separate from production tracking."""

import time
from functools import lru_cache

import cv2
import numpy as np

from .geometry import iou
from .object_geometry import quad_bounds
from .objects import ObjectTracker


@lru_cache(maxsize=1)
def card_picture():
    rng = np.random.default_rng(915)
    card = rng.integers(35, 210, (90, 120, 3), np.uint8)
    card = cv2.GaussianBlur(card, (3, 3), 0)
    cv2.rectangle(card, (3, 3), (116, 86), (230, 190, 60), 3)
    cv2.circle(card, (27, 29), 16, (50, 70, 235), -1)
    cv2.rectangle(card, (72, 15), (106, 49), (50, 220, 90), -1)
    cv2.putText(card, "FP", (31, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    return card


def rigid_frame(index=0, angle=0, hidden=False, twin=False, changed=False):
    image = np.full((360, 640, 3), 22, np.uint8)
    card = card_picture()
    if changed:
        card = np.full_like(card, (240, 30, 200))
    center = (220 + 18 * np.sin(index / 20), 180 + 10 * np.sin(index / 17))

    def paste(cx, cy):
        matrix = cv2.getRotationMatrix2D((60, 45), -angle, 1)
        matrix[:, 2] += np.float64([cx - 60, cy - 45])
        points = cv2.transform(np.float32([[[0, 0], [120, 0], [120, 90], [0, 90]]]), matrix)[0]
        warped = cv2.warpAffine(card, matrix, (640, 360))
        mask = cv2.warpAffine(np.full((90, 120), 255, np.uint8), matrix, (640, 360), flags=cv2.INTER_NEAREST)
        image[mask != 0] = warped[mask != 0]
        return quad_bounds(points / [640, 360])

    expected = paste(*center)
    if hidden:
        image[:] = 22
    if twin:
        paste(center[0] + 125, center[1])
    return image, expected


def validate_tracking():
    results = {}
    for name in ("rotation", "short_occlusion", "lookalikes", "long_occlusion", "changed_appearance"):
        tracker = ObjectTracker()
        timings, overlap, states = [], [], {}
        confirmed = false_confirmations = 0
        reacquired_at = None
        for index in range(90):
            hidden = (
                20 <= index < (25 if name == "short_occlusion" else 66)
                if name in ("short_occlusion", "long_occlusion")
                else index == 20
                if name == "lookalikes"
                else False
            )
            twin = name == "lookalikes" and index > 20
            image, expected = rigid_frame(
                index if name != "lookalikes" else 0,
                angle=index if name == "rotation" else 0,
                hidden=hidden,
                twin=twin,
                changed=name == "changed_appearance" and index >= 20,
            )
            start = time.perf_counter()
            found, _ = tracker.update(image, index / 30, expected if index == 0 else None)
            timings.append((time.perf_counter() - start) * 1000)
            state = tracker.metrics["state"]
            states[state] = states.get(state, 0) + 1
            confirmed += found is not None
            unsafe = hidden or (
                name in ("lookalikes", "long_occlusion", "changed_appearance") and index >= 20
            )
            false_confirmations += int(unsafe and found is not None)
            if found is not None:
                overlap.append(iou(found, expected))
                if name == "short_occlusion" and index >= 25 and reacquired_at is None:
                    reacquired_at = index
        passed = false_confirmations == 0 and (min(overlap) if overlap else 0) >= 0.85
        if name == "rotation":
            passed &= confirmed >= 88
        if name == "short_occlusion":
            passed &= reacquired_at is not None and reacquired_at <= 27
        if name == "lookalikes":
            passed &= states.get("ambiguous", 0) > 0
        results[name] = {
            "passed": bool(passed),
            "frames": 90,
            "confirmed": confirmed,
            "false_confirmations": false_confirmations,
            "minimum_iou_when_confirmed": round(min(overlap), 4) if overlap else None,
            "reacquired_frame": reacquired_at,
            "states": states,
            "mean_tracking_ms": round(float(np.mean(timings)), 2),
            "p95_tracking_ms": round(float(np.percentile(timings, 95)), 2),
        }
    return {
        "passed": all(case["passed"] for case in results.values()),
        "cases": results,
        "scope": "Synthetic rigid 2-D textured card with known bounds, not real-user video. Timing includes tracker only; no decoding/rendering/OBS/Android.",
    }
