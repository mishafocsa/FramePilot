"""Geometric frame candidates; no inference about the meaning of an application's UI."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Rect, iou


@dataclass(frozen=True)
class Slot:
    rect: Rect
    shape: str = "rectangle"


def detect_slots(bgr, roi=None, sensitivity="normal"):
    height, width = bgr.shape[:2]
    region = roi.clipped() if roi is not None else Rect(0, 0, 1, 1)
    x, y = round(region.x * width), round(region.y * height)
    right = min(width, round((region.x + region.w) * width))
    bottom = min(height, round((region.y + region.h) * height))
    if right - x < 8 or bottom - y < 8:
        return []
    image = bgr[y:bottom, x:right]
    ratio = min(1, 1280 / max(image.shape[:2]))
    small = cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))))
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # A 5×5 blur erased one-pixel and dashed UI borders before edge extraction.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 35, 100)
    enhanced = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(blurred)
    weak = cv2.Canny(enhanced, 8, 24)
    variants = [edges, weak]
    # Close short gaps along the axes, without filling the frame's interior.
    gap = 13 if sensitivity == "sensitive" else 9
    joined = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, np.ones((gap, 1), np.uint8))
    variants.append(joined)
    ranked = []
    for edges_ in variants:
        contours, _ = cv2.findContours(edges_, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = abs(cv2.contourArea(contour))
            if not 0.012 * sw * sh < area < 0.98 * sw * sh:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if min(bw, bh) < 0.06 * min(sw, sh):
                continue
            # Toolbars/text fields must not become the default video frame.
            # An explicit search region permits unusually narrow intended frames.
            if max(bw / bh, bh / bw) > (32 if roi is not None else 8):
                continue
            if bx <= 2 and by <= 2 and bx + bw >= sw - 2 and by + bh >= sh - 2:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
            hull = max(1, cv2.contourArea(cv2.convexHull(contour)))
            shape = None
            if 4 <= len(polygon) <= 12 and area / (bw * bh) > 0.83 and area / hull > 0.94:
                shape = "rectangle"
            elif len(contour) >= 12:
                _, (ew, eh), angle = cv2.fitEllipse(contour)
                ellipse_area = np.pi * ew * eh / 4
                aligned = min(abs(angle % 90), abs(90 - angle % 90)) < 8 or abs(ew - eh) < ew * 0.1
                if aligned and ellipse_area > 0 and abs(area / ellipse_area - 1) < 0.08:
                    shape = "ellipse"
            if shape is None:
                continue
            rect = Rect(
                (x + bx / sw * (right - x)) / width,
                (y + by / sh * (bottom - y)) / height,
                bw / sw * (right - x) / width,
                bh / sh * (bottom - y) / height,
            ).clipped()
            if roi is not None and not roi.contains(rect.center):
                continue
            if any(iou(rect, slot.rect) > 0.8 for _, slot in ranked):
                continue
            score = rect.area
            ranked.append((score, Slot(rect, shape)))
    return [slot for _, slot in sorted(ranked, key=lambda item: item[0], reverse=True)[:12]]
