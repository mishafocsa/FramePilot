"""Explicit, deterministic framing rules. They do not infer a third-party app's intent."""

from dataclasses import asdict, dataclass, replace

from .geometry import Rect

WHOLE = Rect(0, 0, 1, 1)


@dataclass(frozen=True)
class FramingRule:
    enabled: bool = False
    whole_when: str = "larger"
    threshold: float = 0.2
    hysteresis: float = 0.1
    transition: float = 0.12

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, value):
        return cls(**value) if value else cls()


class AutoFraming:
    def __init__(self):
        self.current = WHOLE
        self.start = WHOLE
        self.goal = WHOLE
        self.whole = None
        self.started = 0.0
        self.duration = 0.0
        self.configuration = None
        self.clock = None
        self.active = False

    def sample(self, now):
        progress = min(1.0, max(0.0, (now - self.started) / self.duration)) if self.duration > 0 else 1.0
        # Smoothstep gives zero velocity at each end, including a reversed transition.
        amount = progress * progress * (3 - 2 * progress)
        self.current = Rect(
            *(
                a + (b - a) * amount
                for a, b in zip(
                    (self.start.x, self.start.y, self.start.w, self.start.h),
                    (self.goal.x, self.goal.y, self.goal.w, self.goal.h),
                    strict=True,
                )
            )
        )
        self.active = progress < 1 and self.current != self.goal
        return self.current

    def update(self, part: Rect, area: float, rule: FramingRule, now: float):
        if not rule.enabled or part == WHOLE:
            self.current = self.start = self.goal = part
            self.configuration, self.whole, self.clock = None, None, now
            self.active = False
            return part
        # A seek/restart resets animation time; no future state may leak backwards.
        if self.clock is not None and now < self.clock:
            self.configuration, self.whole = None, None
        previous = self.sample(now) if self.configuration is not None else part
        self.clock = now
        configuration = (part, rule)
        if self.configuration != configuration:
            self.whole = None
        low = max(0.0, rule.threshold * (1 - rule.hysteresis))
        high = min(1.0, rule.threshold * (1 + rule.hysteresis))
        if self.whole is None:
            whole = area >= high if rule.whole_when == "larger" else area <= low
        elif rule.whole_when == "larger":
            whole = area >= low if self.whole else area >= high
        else:
            whole = area <= high if self.whole else area <= low
        goal = WHOLE if whole else part
        changed = self.configuration != configuration or goal != self.goal
        first = self.configuration is None
        self.configuration, self.whole = configuration, whole
        if changed:
            self.start, self.goal = previous, goal
            self.started = now
            # Initial state is fully resolved so offline exports have no artificial lead-in.
            self.duration = 0 if first else rule.transition
        return self.sample(now)

    def status(self, rule: FramingRule, area: float):
        return {
            "enabled": rule.enabled,
            "whole": self.whole if self.whole is not None else self.goal == WHOLE,
            "transitioning": self.active,
            "region": [self.current.x, self.current.y, self.current.w, self.current.h],
            "goal_region": [self.goal.x, self.goal.y, self.goal.w, self.goal.h],
            "frame_area": area,
            "whole_when": rule.whole_when,
            "rule": rule.record(),
            "part": (
                [
                    self.configuration[0].x,
                    self.configuration[0].y,
                    self.configuration[0].w,
                    self.configuration[0].h,
                ]
                if self.configuration
                else None
            ),
        }


def freeze_framing(settings, tracks):
    """Freeze the resolved view for export; external-window history is not a video track."""

    def resolved(configuration, state):
        rule = configuration.framing
        if not rule.enabled or configuration.subject_region == WHOLE:
            return {}
        engine = AutoFraming()
        part = configuration.subject_region
        if state.get("rule") == rule.record() and state.get("part") == [part.x, part.y, part.w, part.h]:
            engine.configuration = (part, rule)
            engine.whole = state.get("whole")
        engine.update(part, configuration.target.area, rule, 0)
        return {
            "subject_region": engine.goal,
            "framing": replace(rule, enabled=False),
            "whole_fit": engine.whole is True or engine.goal == WHOLE,
        }

    if settings.slots:
        slots = tuple(
            replace(slot, **resolved(slot, tracks.get(slot.identifier, {}).get("framing", {})))
            for slot in settings.slots
        )
        return replace(settings, slots=slots, framing=replace(settings.framing, enabled=False))
    return replace(settings, **resolved(settings, tracks.get("single", {}).get("framing", {})))
