"""Source/output continuity; no pixel-change heuristic (a static scene is valid video)."""

import math
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SignalPolicy:
    timeout: float = 2.0
    fallback: str = "black"

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, value):
        if not isinstance(value, dict):
            return cls()
        timeout, fallback = value.get("timeout", 2.0), value.get("fallback", "black")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0.5 <= timeout <= 15:
            return cls()
        if fallback not in ("black", "freeze"):
            return cls()
        return cls(float(timeout), fallback)


class SignalMonitor:
    def __init__(self):
        self.armed = self.awaiting = self.expected = self.suspended = False
        self.reference = time.monotonic()
        self.last_input = self.last_result = None
        self.input_stamp = self.result_stamp = None
        self.fault = None

    def arm(self, now=None):
        self.armed = self.awaiting = True
        self.reference = time.monotonic() if now is None else now
        self.last_input = self.last_result = None
        self.input_stamp = self.result_stamp = None
        self.fault = None

    def received(self, stamp, now=None):
        if stamp == self.input_stamp:
            return
        self.input_stamp = stamp
        self.last_input = time.monotonic() if now is None else now

    def complete(self, stamp, submitted_at, timeout, now=None, repaint=False):
        now = time.monotonic() if now is None else now
        if not self.armed or repaint or stamp == self.result_stamp or now - submitted_at > timeout:
            return False
        self.result_stamp = stamp
        self.last_result = now
        self.awaiting = False
        self.fault = None
        return True

    def fail(self, reason):
        if reason not in ("source_error", "processing_error", "source_stopped"):
            raise ValueError("Unknown signal failure")
        self.armed = True
        self.awaiting = False
        self.fault = reason

    def poll(self, expected, timeout, suspended=False, now=None):
        now = time.monotonic() if now is None else now
        if (expected and not self.expected) or (self.suspended and not suspended):
            self.reference = now
            self.last_input = self.last_result = None
        self.expected, self.suspended = expected, suspended
        if suspended:
            state = "calibrating"
        elif not self.armed:
            state = "idle"
        else:
            if not self.fault and (expected or self.awaiting):
                if now - (self.last_input if self.last_input is not None else self.reference) > timeout:
                    self.fault = "input_stalled"
                elif now - (self.last_result if self.last_result is not None else self.reference) > timeout:
                    self.fault = "processing_stalled"
            state = self.fault or ("waiting" if self.awaiting else "live" if expected else "paused")
        return {
            "state": state,
            "blocked": state
            in (
                "waiting",
                "input_stalled",
                "processing_stalled",
                "source_error",
                "processing_error",
                "source_stopped",
            ),
            "input_age_ms": round(max(0, now - self.last_input) * 1000)
            if self.last_input is not None
            else None,
            "result_age_ms": round(max(0, now - self.last_result) * 1000)
            if self.last_result is not None
            else None,
        }
