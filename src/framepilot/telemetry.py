"""Local pipeline counters, deliberately separate from receiver/camera measurements."""

import time
from collections import deque

import numpy as np


class FrameTelemetry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.started = time.monotonic()
        self.received = self.accepted = self.busy_skips = self.throttled = self.completed = self.lost = 0
        self.repaints = 0
        self.processing = deque(maxlen=600)
        self.timestamps = deque(maxlen=600)

    def complete(self, milliseconds, found, now=None, repaint=False):
        if repaint:
            self.repaints += 1
        else:
            self.completed += 1
            self.lost += not found
        self.processing.append(float(milliseconds))
        if not repaint:
            self.timestamps.append(time.monotonic() if now is None else now)

    def summary(self, now=None):
        now = time.monotonic() if now is None else now
        recent = [stamp for stamp in self.timestamps if now - 5 <= stamp <= now]
        span = recent[-1] - recent[0] if len(recent) > 1 else 0
        return {
            "input_callbacks": self.received,
            "submitted_frames": self.accepted,
            "skipped_while_busy": self.busy_skips,
            "skipped_by_fps_limit": self.throttled,
            "completed_frames": self.completed,
            "repaint_jobs": self.repaints,
            "unconfirmed_target_frames": self.lost,
            "recent_processing_fps": round((len(recent) - 1) / span, 2) if span > 0 else None,
            "mean_worker_ms": round(float(np.mean(self.processing)), 2) if self.processing else None,
            "p95_worker_ms": round(float(np.percentile(self.processing, 95)), 2) if self.processing else None,
            "scope": "Local accepted pipeline jobs; includes local JPEG if output enabled, excludes receiver; unconfirmed is not ground-truth accuracy",
        }
