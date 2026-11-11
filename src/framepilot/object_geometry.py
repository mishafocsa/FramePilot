"""Geometry for a selected rigid 2-D patch; no object class or identity inference."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Rect
from .vision import appearance


@dataclass(frozen=True)
class QuadSelection:
    quad: tuple[tuple[float, float], ...]


def inverse_region(quad, selection):
    if not valid_quad(quad):
        raise ValueError("Нет подтверждённой поверхности")
    mapping = cv2.getPerspectiveTransform(np.float32(quad), rectangle_quad(Rect(0, 0, 1, 1)))
    points = cv2.perspectiveTransform(np.float32([rectangle_quad(selection)]), mapping)[0]
    return quad_bounds(points).clipped()


def rectangle_quad(rect):
    return np.float32(
        [
            [rect.x, rect.y],
            [rect.x + rect.w, rect.y],
            [rect.x + rect.w, rect.y + rect.h],
            [rect.x, rect.y + rect.h],
        ]
    )


def valid_quad(value):
    try:
        points = np.asarray(value, np.float32)
        return (
            points.shape == (4, 2)
            and bool(np.isfinite(points).all())
            and bool((points >= -0.1).all() and (points <= 1.1).all())
            and cv2.isContourConvex(points)
            and cv2.contourArea(points) > 1e-7
        )
    except (ValueError, TypeError, cv2.error):
        return False


def region_quad(quad, region):
    """Map a region of the original selection into its current oriented footprint."""
    if not valid_quad(quad):
        raise ValueError("Нет подтверждённой геометрии поверхности")
    matrix = cv2.getPerspectiveTransform(rectangle_quad(Rect(0, 0, 1, 1)), np.float32(quad))
    return cv2.perspectiveTransform(np.float32([rectangle_quad(region)]), matrix)[0]


def quad_bounds(quad):
    low, high = quad.min(axis=0), quad.max(axis=0)
    return Rect(float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1]))


def canonical_patch(image, quad, size):
    h, w = image.shape[:2]
    tw, th = size
    destination = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    matrix = cv2.getPerspectiveTransform(np.float32(quad * [w, h]), destination)
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR)


def measure_patch(image, quad, reference, histogram):
    patch = canonical_patch(image, quad, (reference.shape[1], reference.shape[0]))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    error = float(cv2.matchTemplate(gray, reference, cv2.TM_SQDIFF_NORMED)[0, 0])
    hist = appearance(patch, Rect(0, 0, 1, 1))
    distance = float(cv2.compareHist(histogram, hist, cv2.HISTCMP_BHATTACHARYYA)) if hist is not None else 1.0
    return error, distance


def refine_quad(image, quad, reference):
    """Bounded registration against the fixed reference to limit cumulative flow drift."""
    size = reference.shape[1], reference.shape[0]
    sample = cv2.cvtColor(canonical_patch(image, quad, size), cv2.COLOR_BGR2GRAY)
    matrix = np.eye(2, 3, dtype=np.float32)
    try:
        _, matrix = cv2.findTransformECC(
            reference,
            sample,
            matrix,
            cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 15, 0.001),
            None,
            3,
        )
    except cv2.error:
        return quad
    if abs(float(np.arctan2(matrix[1, 0], matrix[0, 0]))) > 0.1 or np.linalg.norm(matrix[:, 2]) > 5:
        return quad
    w, h = size
    canonical = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    adjusted = cv2.transform(canonical, matrix)
    mapping = cv2.getPerspectiveTransform(canonical[0], np.float32(quad))
    return cv2.perspectiveTransform(adjusted, mapping)[0]


def search_pattern(template, quad, width, height, factor):
    pixels = quad * [width, height]
    across, down = pixels[1] - pixels[0], pixels[3] - pixels[0]
    tw, th = max(8, round(np.linalg.norm(across) * factor)), max(8, round(np.linalg.norm(down) * factor))
    angle = float(np.degrees(np.arctan2(across[1], across[0])))
    matrix = cv2.getRotationMatrix2D((0, 0), -angle, 1)
    corners = cv2.transform(np.float32([[[0, 0], [tw, 0], [tw, th], [0, th]]]), matrix)[0]
    low, high = corners.min(axis=0), corners.max(axis=0)
    matrix[:, 2] -= low
    size = tuple(int(max(8, np.ceil(value))) for value in high - low)
    patch = cv2.resize(template, (tw, th))
    pattern = cv2.warpAffine(patch, matrix, size)
    mask = cv2.warpAffine(np.full((th, tw), 255, np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
    return pattern, mask, np.float32(corners - low)


def match_candidates(gray, template, quad, image_size, factors, origin=(0, 0)):
    """Up to two spatially separate peaks per scale, then NMS across scales."""
    from .geometry import iou

    width, height = image_size
    ranked = []
    for factor in factors:
        pattern, mask, corners = search_pattern(template, quad, width, height, factor)
        ph, pw = pattern.shape
        if pw > gray.shape[1] or ph > gray.shape[0]:
            continue
        scores = cv2.matchTemplate(gray, pattern, cv2.TM_SQDIFF_NORMED, mask=mask)
        scores[~np.isfinite(scores)] = 1
        for _ in range(2):
            error, _, point, _ = cv2.minMaxLoc(scores)
            if error > 0.2:
                break
            candidate_quad = (corners + np.float32([origin[0] + point[0], origin[1] + point[1]])) / [
                width,
                height,
            ]
            ranked.append((error, quad_bounds(candidate_quad), np.float32(candidate_quad)))
            cv2.rectangle(
                scores,
                (max(0, point[0] - pw // 2), max(0, point[1] - ph // 2)),
                (min(scores.shape[1] - 1, point[0] + pw // 2), min(scores.shape[0] - 1, point[1] + ph // 2)),
                1,
                -1,
            )
    distinct = []
    for item in sorted(ranked, key=lambda value: value[0]):
        if not any(iou(item[1], previous[1]) > 0.5 for previous in distinct):
            distinct.append(item)
    return distinct
