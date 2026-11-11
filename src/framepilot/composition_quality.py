"""Geometric framing evidence, not semantic object visibility or perceptual image quality."""

import cv2
import numpy as np


def composition_report(focus, placement, destination_alpha, source_size):
    height, width = destination_alpha.shape
    scale = min(1, 256 / max(width, height))
    sw, sh = max(2, round(width * scale)), max(2, round(height * scale))
    sx, sy = sw / width, sh / height
    polygon = np.asarray(focus, np.float32)
    projected = polygon * placement.scale + [placement.x, placement.y]
    normalized = np.float32(projected * [sx, sy])
    expected = abs(cv2.contourArea(normalized))
    region = np.zeros((sh, sw), np.uint8)
    cv2.fillConvexPoly(region, np.round(normalized).astype(np.int32), 255)
    allowed = cv2.resize(destination_alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    iw, ih = source_size
    source = np.float32([[[0, 0], [iw, 0], [iw, ih], [0, ih]]])
    source = (source * placement.scale + [placement.x, placement.y]) * [sx, sy]
    available = np.zeros_like(region)
    cv2.fillConvexPoly(available, np.round(source[0]).astype(np.int32), 255)
    inside = region > 0
    visible = float(np.sum(allowed[inside] * (available[inside] / 255)))
    # A rasterized polygon contains boundary pixels. Correct its normalization when fully on-canvas.
    bounds = cv2.boundingRect(normalized)
    within = bounds[0] >= 0 and bounds[1] >= 0 and bounds[0] + bounds[2] <= sw and bounds[1] + bounds[3] <= sh
    denominator = float(inside.sum()) if within else max(expected, float(inside.sum()))
    coverage = min(1.0, max(0.0, visible / max(denominator, 1)))
    warnings = []
    if coverage < 0.95:
        warnings.append("Выбранная область обрезается рамкой или краем исходника")
    if placement.scale > 2:
        warnings.append("Сильное увеличение: детализация ограничена исходными пикселями")
    return {
        "state": "warning" if warnings else "good",
        "visible_fraction": round(coverage, 3),
        "upscale": round(float(placement.scale), 2),
        "warnings": warnings,
        "scope": "Estimated visibility of the selected rectangle/polygon; not object segmentation or perceptual quality",
    }
