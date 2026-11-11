"""Immutable clip-specific cues and analysed trajectories. No frame images are persisted."""

import bisect
import hashlib
import math
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

from .geometry import Rect
from .object_geometry import valid_quad


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Неверное число в проекте ролика")
    return float(value)


def rectangle(value, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Неверные координаты в проекте")
    x, y, w, h = [number(v, 0, 1) for v in value]
    if min(w, h) <= 0 or x + w > 1 + 1e-9 or y + h > 1 + 1e-9:
        raise ValueError("Область выходит за пределы изображения")
    return Rect(x, y, w, h)


def rect_record(value):
    return [value.x, value.y, value.w, value.h] if value is not None else None


def source_identity(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        stream.seek(max(0, stat.st_size - 1024 * 1024))
        digest.update(stream.read(1024 * 1024))
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sample_sha256": digest.hexdigest()}


@dataclass(frozen=True)
class Cue:
    time: float
    slot: str
    subject: Rect | None
    region: Rect
    padding: float = 0.25
    zoom: float = 1.0
    quad: tuple[tuple[float, float], ...] | None = None

    def record(self):
        return {
            "time": self.time,
            "slot": self.slot,
            "subject": rect_record(self.subject),
            "region": rect_record(self.region),
            "padding": self.padding,
            "zoom": self.zoom,
            "quad": [list(p) for p in self.quad] if self.quad is not None else None,
        }

    @classmethod
    def from_record(cls, data):
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("slot"), str)
            or not 1 <= len(data["slot"]) <= 80
        ):
            raise ValueError("Неверная поправка")
        quad = data.get("quad")
        if quad is not None and not valid_quad(quad):
            raise ValueError("Неверная геометрия поправки")
        return cls(
            number(data.get("time"), 0, 86400),
            data["slot"],
            rectangle(data.get("subject"), True),
            rectangle(data.get("region")),
            number(data.get("padding"), 0, 1),
            number(data.get("zoom"), 0.5, 2),
            tuple(tuple(float(v) for v in p) for p in quad) if quad is not None else None,
        )


@dataclass(frozen=True)
class Sample:
    subject: Rect | None
    state: str = "stable"
    eyes: tuple[float, float] | None = None
    quad: tuple[tuple[float, float], ...] | None = None

    def record(self):
        return [
            rect_record(self.subject),
            self.state,
            list(self.eyes) if self.eyes is not None else None,
            [list(p) for p in self.quad] if self.quad is not None else None,
        ]

    @classmethod
    def from_record(cls, value):
        if (
            not isinstance(value, list)
            or len(value) not in (3, 4)
            or value[1] not in ("stable", "weak", "lost", "ambiguous", "recovering", "selected")
        ):
            raise ValueError("Неверный кадр анализа")
        eyes = value[2]
        if eyes is not None:
            if not isinstance(eyes, list) or len(eyes) != 2:
                raise ValueError("Неверные ориентиры глаз")
            eyes = tuple(number(v, 0, 1) for v in eyes)
        quad = value[3] if len(value) == 4 else None
        if quad is not None and not valid_quad(quad):
            raise ValueError("Неверная геометрия поверхности в анализе")
        return cls(
            rectangle(value[0], True),
            value[1],
            eyes,
            tuple(tuple(float(v) for v in p) for p in quad) if quad is not None else None,
        )


@dataclass(frozen=True)
class Trajectory:
    slot: str
    mode: str
    start: float
    fps: int
    samples: tuple[Sample, ...]

    def at(self, now):
        position = (now - self.start) * self.fps
        if position < -0.01 or position >= len(self.samples):
            return False, None
        index = max(0, int(position))
        a = self.samples[index]
        if index + 1 == len(self.samples):
            return True, a
        b = self.samples[index + 1]
        if a.subject is None or b.subject is None:
            return True, a
        fraction = position - index
        values = [x + (y - x) * fraction for x, y in zip(rect_record(a.subject), rect_record(b.subject))]
        eyes = tuple(x + (y - x) * fraction for x, y in zip(a.eyes, b.eyes)) if a.eyes and b.eyes else a.eyes
        quad = (
            tuple(tuple(x + (y - x) * fraction for x, y in zip(p, q)) for p, q in zip(a.quad, b.quad))
            if a.quad is not None and b.quad is not None
            else a.quad
        )
        if quad is not None and not valid_quad(quad):
            quad = None
        return True, Sample(Rect(*values), a.state, eyes, quad)

    def problems(self):
        spans = []
        for index, sample in enumerate(self.samples):
            state = "lost" if sample.subject is None else sample.state
            if state in ("stable", "selected"):
                continue
            start, end = self.start + index / self.fps, self.start + (index + 1) / self.fps
            if spans and spans[-1][2] == state and abs(spans[-1][1] - start) < 1e-6:
                spans[-1] = spans[-1][0], end, state
            else:
                spans.append((start, end, state))
        return spans

    def record(self):
        return {
            "slot": self.slot,
            "mode": self.mode,
            "start": self.start,
            "fps": self.fps,
            "samples": [s.record() for s in self.samples],
        }

    @classmethod
    def from_record(cls, value):
        if (
            not isinstance(value, dict)
            or value.get("mode") not in ("face", "object", "upper", "person")
            or not isinstance(value.get("slot"), str)
            or not 1 <= len(value["slot"]) <= 80
        ):
            raise ValueError("Неверная траектория")
        samples = value.get("samples")
        if (
            not isinstance(samples, list)
            or not 1 <= len(samples) <= 200000
            or type(value.get("fps")) is not int
            or value["fps"] not in (15, 24, 30)
        ):
            raise ValueError("Неверная длительность или частота анализа")
        return cls(
            value["slot"],
            value["mode"],
            number(value.get("start"), 0, 86400),
            value["fps"],
            tuple(Sample.from_record(s) for s in samples),
        )


@dataclass(frozen=True)
class Automation:
    cues: tuple[Cue, ...] = ()
    tracks: tuple[Trajectory, ...] = ()
    enabled: bool = True
    override_time: float | None = None  # Transient preview edit; never stored in clip projects.
    override_slot: str = "single"

    @cached_property
    def cue_index(self):
        result = {}
        for slot in {c.slot for c in self.cues}:
            keys = tuple(sorted((c for c in self.cues if c.slot == slot), key=lambda c: c.time))
            result[slot] = keys, tuple(c.time for c in keys)
        return result

    def keys(self, slot):
        return self.cue_index.get(slot, ((), ()))[0]

    def cue(self, slot, now):
        keys, times = self.cue_index.get(slot, ((), ()))
        index = bisect.bisect_right(times, now + 1e-6) - 1
        return keys[index] if index >= 0 else None

    def settings_at(self, settings, now):
        if not self.enabled or (
            self.override_time is not None
            and self.override_slot == settings.automation_slot
            and abs(now - self.override_time) < 1e-5
        ):
            return settings
        keys, times = self.cue_index.get(settings.automation_slot, ((), ()))
        index = bisect.bisect_right(times, now + 1e-6) - 1
        if index < 0:
            return settings
        a = keys[index]
        b = keys[min(index + 1, len(keys) - 1)]
        t = max(0, min(1, (now - a.time) / max(b.time - a.time, 1e-9))) if a != b else 0
        t = t * t * (3 - 2 * t)
        region = Rect(*(x + (y - x) * t for x, y in zip(rect_record(a.region), rect_record(b.region))))
        return replace(
            settings,
            subject_region=region,
            padding=a.padding + (b.padding - a.padding) * t,
            framing_zoom=a.zoom + (b.zoom - a.zoom) * t,
        )

    def cached(self, slot, mode, now):
        if self.enabled:
            for track in self.tracks:
                if track.slot == slot and track.mode == mode:
                    return track.at(now)
        return False, None

    def record(self):
        return {
            "enabled": self.enabled,
            "cues": [c.record() for c in self.cues],
            "tracks": [t.record() for t in self.tracks],
        }

    @classmethod
    def from_record(cls, value):
        if (
            not isinstance(value, dict)
            or type(value.get("enabled")) is not bool
            or not isinstance(value.get("cues"), list)
            or not isinstance(value.get("tracks"), list)
            or len(value["cues"]) > 5000
            or len(value["tracks"]) > 4
        ):
            raise ValueError("Неверный проект анализа")
        cues = tuple(Cue.from_record(c) for c in value["cues"])
        tracks = tuple(Trajectory.from_record(t) for t in value["tracks"])
        if (
            len({(c.slot, c.time) for c in cues}) != len(cues)
            or len({t.slot for t in tracks}) != len(tracks)
            or sum(len(t.samples) for t in tracks) > 200000
        ):
            raise ValueError("Дублированные поправки или слишком большой анализ")
        return cls(cues, tracks, value["enabled"])
