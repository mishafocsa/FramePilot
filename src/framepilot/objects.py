"""Local template tracking for manually selected objects. No semantic/identity recognition."""

import cv2
import numpy as np

from .geometry import Rect, iou
from .object_geometry import (
    QuadSelection,
    canonical_patch,
    match_candidates,
    measure_patch,
    quad_bounds,
    rectangle_quad,
    refine_quad,
    valid_quad,
)
from .vision import appearance


class ObjectTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.box = None
        self.quad = None
        self.template = None
        self.histogram = None
        self.last_seen = -1e9
        self.missing_since = None
        self.proposals = []
        self.last_search = -1e9
        self.gray = None
        self.points = None
        self.verification = None
        self.flow_updates = 0
        self.template_updates = 0
        self.backend = "ожидание выделения"
        self.pending = None
        self.pending_at = None
        self.last_update = None
        self.metrics = {"state": "unselected", "basis": "none"}

    def diagnostics(self):
        return {
            **self.metrics,
            "basis": self.backend,
            "scope": "Heuristic tracking evidence, not calibrated identity probability",
        }

    @staticmethod
    def flow_image(gray):
        h, w = gray.shape
        scale = min(1, 640 / max(w, h))
        return cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else gray

    def remember(self, gray, points=None):
        self.gray = self.flow_image(gray)
        if points is not None and len(points) >= 12 and self.flow_updates % 6:
            self.points = points
            return
        h, w = self.gray.shape
        r = self.box.pixels(w, h)
        x, y = max(0, round(r.x)), max(0, round(r.y))
        x2, y2 = min(w, round(r.x + r.w)), min(h, round(r.y + r.h))
        patch = self.gray[y:y2, x:x2]
        self.points = None
        if min(patch.shape, default=0) < 8:
            return
        mask = np.zeros(patch.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(self.quad * [w, h] - [x, y]).astype(np.int32), 255)
        points = cv2.goodFeaturesToTrack(
            patch,
            maxCorners=80,
            qualityLevel=0.02,
            minDistance=max(3, min(patch.shape) / 16),
            blockSize=3,
            mask=mask,
        )
        if points is not None:
            self.points = points + np.float32([[[x, y]]])

    def flow(self, image, gray):
        if self.gray is None or self.points is None or len(self.points) < 8 or self.missing_since is not None:
            return None
        current = self.flow_image(gray)
        if current.shape != self.gray.shape:
            return None
        options = dict(
            winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
        forward, status, error = cv2.calcOpticalFlowPyrLK(self.gray, current, self.points, None, **options)
        if forward is None or status is None:
            return None
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, self.gray, forward, None, **options)
        if backward is None or reverse_status is None:
            return None
        consistent = np.linalg.norm(backward - self.points, axis=2).ravel() < 1.5
        valid = (status.ravel() == 1) & (reverse_status.ravel() == 1) & consistent
        if error is not None:
            valid &= error.ravel() < 30
        if int(valid.sum()) < 8:
            return None
        old, new = self.points[valid], forward[valid]
        matrix, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=200
        )
        if matrix is None or inliers is None or float(inliers.mean()) < 0.65:
            return None
        factor = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not 0.8 <= factor <= 1.25 or abs(float(np.arctan2(matrix[1, 0], matrix[0, 0]))) > 0.35:
            return None
        h, w = current.shape
        quad = cv2.transform(np.float32([self.quad * [w, h]]), matrix)[0] / [w, h]
        candidate = quad_bounds(quad)
        if candidate.clipped().area < candidate.area * 0.98:
            return None
        # Compare in the ORIGINAL selection's coordinates, including cumulative rotation.
        score, distance = measure_patch(image, quad, self.verification, self.histogram)
        if self.flow_updates % 8 == 0:
            refined = refine_quad(image, quad, self.verification)
            refined_score, refined_distance = measure_patch(image, refined, self.verification, self.histogram)
            if refined_score < score and refined_distance <= 0.45:
                quad, candidate, score, distance = (
                    refined,
                    quad_bounds(refined),
                    refined_score,
                    refined_distance,
                )
                if candidate.clipped().area < candidate.area * 0.98:
                    return None
        if score > 0.16 or distance > 0.45:
            return None
        good = new[inliers.ravel() != 0]
        return candidate, good, np.float32(quad), score, distance, float(inliers.mean())

    def search(self, image, gray, now):
        if now - self.last_search < 0.8:
            return
        self.last_search = now
        self.proposals = []
        h, w = gray.shape
        candidates = match_candidates(gray, self.template, self.quad, (w, h), (0.7, 0.85, 1, 1.2, 1.4))
        for _, rect, quad in candidates:
            score, distance = measure_patch(image, quad, self.verification, self.histogram)
            if score <= 0.2 and distance < 0.5:
                self.proposals.append(rect)
            if len(self.proposals) == 3:
                break

    def update(self, image, now, selection=None):
        if selection is None and now == self.last_update:
            return (self.box if self.missing_since is None else None), []
        self.last_update = now
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if isinstance(selection, (Rect, QuadSelection)):
            if isinstance(selection, QuadSelection):
                if not valid_quad(selection.quad):
                    raise ValueError("Неверная геометрия выбранной поверхности")
                quad = np.float32(selection.quad)
                pixels = quad * [w, h]
                tw, th = (
                    round(np.linalg.norm(pixels[1] - pixels[0])),
                    round(np.linalg.norm(pixels[3] - pixels[0])),
                )
                if min(tw, th) < 8 or max(tw, th) > max(w, h) * 2:
                    raise ValueError("Поверхность слишком мала или выходит за пределы кадра")
                self.template = cv2.cvtColor(canonical_patch(image, quad, (tw, th)), cv2.COLOR_BGR2GRAY)
                self.box = quad_bounds(quad)
                self.quad = quad
            else:
                r = selection.clipped().pixels(w, h)
                x, y = max(0, int(r.x)), max(0, int(r.y))
                x2, y2 = min(w, int(r.x + r.w)), min(h, int(r.y + r.h))
                if x2 - x < 8 or y2 - y < 8:
                    raise ValueError("Выделите предмет размером хотя бы 8×8 пикселей")
                self.template = gray[y:y2, x:x2].copy()
                self.box = Rect(x / w, y / h, (x2 - x) / w, (y2 - y) / h)
                self.quad = rectangle_quad(self.box)
            scale = min(1, 96 / max(self.template.shape))
            self.verification = cv2.resize(
                self.template,
                (
                    max(8, round(self.template.shape[1] * scale)),
                    max(8, round(self.template.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            patch = canonical_patch(
                image, self.quad, (self.verification.shape[1], self.verification.shape[0])
            )
            self.histogram = appearance(patch, Rect(0, 0, 1, 1))
            self.last_seen = now
            self.missing_since = None
            self.proposals = []
            self.pending = self.pending_at = None
            self.remember(gray)
            self.backend = "выделение"
            self.metrics = {
                "state": "selected",
                "points": len(self.points) if self.points is not None else 0,
                "template_error": 0.0,
            }
            return self.box, []
        if (
            self.box is None
            or self.template is None
            or (self.missing_since is not None and now - self.missing_since > 1.2)
        ):
            if self.box is not None and self.template is not None:
                self.search(image, gray, now)
            self.backend = "поиск кандидатов"
            self.metrics = {
                "state": "lost" if self.box is not None else "unselected",
                "candidates": len(self.proposals),
            }
            return None, []
        tracked = self.flow(image, gray)
        if tracked is not None:
            self.box, points, self.quad, score, distance, ratio = tracked
            self.last_seen, self.missing_since = now, None
            self.proposals = []
            self.flow_updates += 1
            self.backend = "движение + проверка образца"
            self.metrics = {
                "state": "stable" if len(points) >= 12 and ratio >= 0.8 else "weak",
                "points": len(points),
                "inlier_ratio": round(ratio, 3),
                "template_error": round(score, 4),
                "appearance_distance": round(distance, 4),
                "rotation_degrees": round(
                    float(np.degrees(np.arctan2(*((self.quad[1] - self.quad[0]) * [w, h])[::-1]))), 1
                ),
            }
            self.remember(gray, points)
            return self.box, []
        r = self.box.pixels(w, h)
        margin_x, margin_y = max(24, 1.5 * r.w), max(24, 1.5 * r.h)
        x, y = max(0, int(r.x - margin_x)), max(0, int(r.y - margin_y))
        x2, y2 = min(w, int(r.x + r.w + margin_x)), min(h, int(r.y + r.h + margin_y))
        search = gray[y:y2, x:x2]
        ranked = []
        for _, candidate, quad in match_candidates(
            search, self.template, self.quad, (w, h), (0.9, 0.95, 1, 1.05, 1.1), (x, y)
        ):
            score, distance = measure_patch(image, quad, self.verification, self.histogram)
            if score <= 0.16 and distance <= 0.45:
                ranked.append((score, candidate, quad, distance))
        ranked.sort(key=lambda item: item[0])
        ambiguous = len(ranked) > 1
        if not ranked or ambiguous:
            if self.missing_since is None:
                self.missing_since = now
            self.gray, self.points = None, None
            self.pending = self.pending_at = None
            self.proposals = [row[1] for row in ranked[:3]] if ambiguous else []
            self.backend = "несколько похожих кандидатов" if ambiguous else "цель потеряна"
            self.metrics = {"state": "ambiguous" if ambiguous else "lost", "candidates": len(self.proposals)}
            return None, []
        score, candidate, quad, distance = ranked[0]
        if self.missing_since is not None:
            confirmed = (
                self.pending is not None
                and iou(self.pending, candidate) >= 0.6
                and now - self.pending_at <= 0.5
            )
            self.pending, self.pending_at = candidate, now
            if not confirmed:
                self.backend = "проверка восстановления"
                self.metrics = {"state": "recovering", "template_error": round(score, 4)}
                return None, []
        self.box, self.quad = candidate, quad
        self.last_seen = now
        self.missing_since = None
        self.proposals = []
        self.template_updates += 1
        self.backend = "поиск образца"
        self.pending = self.pending_at = None
        self.metrics = {
            "state": "weak",
            "template_error": round(score, 4),
            "appearance_distance": round(distance, 4),
        }
        self.remember(gray)
        return self.box, []
