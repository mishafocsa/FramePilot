import math

import pytest

from framepilot.geometry import Placement, Rect, StableTarget, fit, iou, smooth


def mapped(rect, p):
    return Rect(rect.x * p.scale + p.x, rect.y * p.scale + p.y, rect.w * p.scale, rect.h * p.scale)


def test_fit_keeps_subject_inside_offset_frame():
    subject, target = Rect(200, 50, 100, 160), Rect(700, 100, 240, 300)
    result = mapped(subject, fit(subject, target, padding=0.15))
    assert result.center == pytest.approx(target.center)
    assert target.contains((result.x, result.y))
    assert target.contains((result.x + result.w, result.y + result.h))


def test_fill_covers_frame():
    subject, target = Rect(0, 0, 100, 200), Rect(100, 50, 300, 100)
    result = mapped(subject, fit(subject, target, padding=0, fill=True))
    assert result.w >= target.w
    assert result.h >= target.h
    assert result.center == target.center


def test_motion_and_zoom_are_bounded():
    result = smooth(Placement(1, 0, 0), Placement(10, 10000, 10000), 0.1)
    assert math.hypot(result.x, result.y) <= 90.001
    assert math.log(result.scale) <= 0.14001


def test_no_movement_when_paused_or_clock_moves_backwards():
    old = Placement(1, 10, 20)
    for dt in (0, -1):
        assert smooth(old, Placement(2, 500, 500), dt) == old


def test_zero_rect_is_rejected():
    with pytest.raises(ValueError):
        fit(Rect(0, 0, 0, 10), Rect(0, 0, 100, 100))


def test_disjoint_candidates_do_not_match():
    assert iou(Rect(0, 0, 0.1, 0.1), Rect(0.8, 0.8, 0.1, 0.1)) == 0


def test_stable_target_requires_confirmation_and_ignores_order():
    stable = StableTarget()
    target, other = Rect(0.1, 0.2, 0.3, 0.4), Rect(0.7, 0.1, 0.2, 0.2)
    assert stable.update([target, other]) is None
    assert stable.update([other, target]) == target
    assert stable.update([]) is None
    assert stable.accepted == target
    assert stable.update([target]) is None
    assert stable.update([target]) == target


def test_changed_slot_requires_two_frames():
    stable = StableTarget()
    old, new = Rect(0, 0, 0.3, 0.3), Rect(0.6, 0.6, 0.3, 0.3)
    stable.update([old])
    stable.update([old])
    assert stable.update([new]) is None
    assert stable.accepted == old
    assert stable.update([new]) == new
