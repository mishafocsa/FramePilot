"""Normalized image coordinates throughout: origin at the TOP LEFT."""

from dataclasses import dataclass
from math import exp, hypot, log


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def contains(self, point: tuple[float, float]) -> bool:
        return self.x <= point[0] <= self.x + self.w and self.y <= point[1] <= self.y + self.h

    def pixels(self, width: int, height: int) -> "Rect":
        return Rect(self.x * width, self.y * height, self.w * width, self.h * height)

    def region(self, relative: "Rect") -> "Rect":
        """Place a normalized subregion inside this rectangle."""
        return Rect(
            self.x + relative.x * self.w,
            self.y + relative.y * self.h,
            relative.w * self.w,
            relative.h * self.h,
        )

    def clipped(self) -> "Rect":
        x, y = max(0.0, min(1.0, self.x)), max(0.0, min(1.0, self.y))
        return Rect(x, y, max(0.0, min(1.0, self.x + self.w) - x), max(0.0, min(1.0, self.y + self.h) - y))


def iou(a: Rect, b: Rect) -> float:
    area = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x)) * max(
        0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    )
    return area / max(a.area + b.area - area, 1e-9)


@dataclass(frozen=True)
class Placement:
    scale: float
    x: float
    y: float


def fit(subject: Rect, target: Rect, padding: float = 0.25, fill: bool = False) -> Placement:
    if subject.area <= 0 or target.area <= 0:
        raise ValueError("Subject and target must have positive dimensions")
    sx = target.w / (subject.w * (1 + 2 * max(0, padding)))
    sy = target.h / (subject.h * (1 + 2 * max(0, padding)))
    scale = max(sx, sy) if fill else min(sx, sy)
    cx, cy = subject.center
    tx, ty = target.center
    return Placement(scale, tx - cx * scale, ty - cy * scale)


def smooth(
    old: Placement | None,
    goal: Placement,
    dt: float,
    response: float = 0.35,
    speed: float = 900,
    zoom_rate: float = 1.4,
) -> Placement:
    if old is None:
        return goal
    dt = max(0, min(dt, 0.15))
    alpha = 1 - exp(-dt / max(response, 0.01))
    dx, dy = (goal.x - old.x) * alpha, (goal.y - old.y) * alpha
    distance = hypot(dx, dy)
    if distance > speed * dt and distance > 0:
        dx, dy = dx * speed * dt / distance, dy * speed * dt / distance
    change = (log(max(goal.scale, 1e-6)) - log(max(old.scale, 1e-6))) * alpha
    change = max(-zoom_rate * dt, min(zoom_rate * dt, change))
    return Placement(old.scale * exp(change), old.x + dx, old.y + dy)


class StableTarget:
    """Require two agreeing detections. Keep last accepted slot through brief failures."""

    def __init__(self) -> None:
        self.accepted: Rect | None = None
        self.pending: Rect | None = None
        self.count = 0

    def update(self, candidates: list[Rect]) -> Rect | None:
        if not candidates:
            self.pending, self.count = None, 0
            return None
        anchor = self.pending or self.accepted
        candidate = max(candidates, key=lambda r: iou(r, anchor)) if anchor else candidates[0]
        if anchor and iou(candidate, anchor) < 0.25:
            candidate = candidates[0]
        self.count = self.count + 1 if self.pending and iou(candidate, self.pending) > 0.65 else 1
        self.pending = candidate
        if self.count >= 2:
            if (
                self.accepted is not None
                and max(
                    abs(candidate.x - self.accepted.x),
                    abs(candidate.y - self.accepted.y),
                    abs(candidate.w - self.accepted.w),
                    abs(candidate.h - self.accepted.h),
                )
                < 0.003
            ):
                return self.accepted
            self.accepted = candidate
            return candidate
        return None
