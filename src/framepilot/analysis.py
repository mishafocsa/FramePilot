"""Cancellable full-frame clip analysis in its own worker, independent of playback."""

import threading
from dataclasses import dataclass, replace
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from .automation import Automation, Sample, Trajectory, source_identity
from .masks import FrameMask
from .pipeline import Pipeline, Settings


@dataclass(frozen=True)
class AnalysisJob:
    source: Path
    settings: Settings
    selection: object
    start: float
    fps: int
    token: int = 0


class AnalysisCancelled(Exception):
    pass


def analyze_clip(job, cancel, progress=lambda _: None, pipeline_factory=Pipeline):
    identity = source_identity(job.source)
    if job.fps not in (15, 24, 30) or job.start < 0:
        raise ValueError("Неверные параметры анализа")
    settings = replace(
        job.settings,
        width=320,
        height=180,
        remove_background=False,
        background_image="",
        frame_mask=FrameMask(),
        eye_level=job.settings.eye_level or 0.38
        if job.settings.subject_mode == "face"
        else job.settings.eye_level,
        automation=Automation(job.settings.automation.cues),
        slots=tuple(
            replace(
                s,
                frame_mask=FrameMask(),
                eye_level=s.eye_level or 0.38 if s.subject_mode == "face" else s.eye_level,
            )
            for s in job.settings.slots
        ),
    )
    tracks = {s.identifier: [] for s in settings.slots} if settings.slots else {"single": []}
    modes = (
        {s.identifier: s.subject_mode for s in settings.slots}
        if settings.slots
        else {"single": settings.subject_mode}
    )
    pipeline = pipeline_factory()
    frames = imageio_ffmpeg.read_frames(
        str(job.source),
        pix_fmt="bgr24",
        input_params=["-ss", f"{job.start:.6f}"],
        output_params=["-vf", f"fps={job.fps}", "-threads", "1"],
    )
    try:
        metadata = next(frames)
        width, height = metadata["size"]
        expected = max(1, (metadata.get("duration", 0) - job.start) * job.fps)
        previous = -1
        import cv2

        for index, data in enumerate(frames):
            if cancel.is_set():
                raise AnalysisCancelled()
            if index * len(tracks) >= 200000:
                raise ValueError("Лимит анализа — 200 000 кадров целей. Разделите длинный ролик.")
            image = np.frombuffer(data, np.uint8).reshape(height, width, 3)
            scale = min(1, 960 / max(width, height))
            if scale < 1:
                image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
            pipeline.process(
                image,
                job.start + index / job.fps,
                settings,
                job.selection if index == 0 else None,
                index == 0,
            )
            for slot in tracks:
                child = pipeline.children[slot] if settings.slots else pipeline
                subject = child.last_face
                state = child.last_quality.get("state", "stable") if subject else "lost"
                if state not in ("stable", "weak", "lost", "ambiguous", "recovering", "selected"):
                    state = "weak" if subject else "lost"
                quad = child.object_tracker.quad if modes[slot] == "object" and subject is not None else None
                tracks[slot].append(
                    Sample(
                        subject,
                        state,
                        child.last_eyes,
                        tuple(tuple(float(v) for v in p) for p in quad) if quad is not None else None,
                    )
                )
            percentage = min(99, round((index + 1) / expected * 100))
            if percentage != previous:
                progress(percentage)
                previous = percentage
        if not any(tracks.values()):
            raise ValueError("После выбранного кадра нет видео")
        if cancel.is_set():
            raise AnalysisCancelled()
        if source_identity(job.source) != identity:
            raise ValueError("Исходный ролик изменился во время анализа")
        progress(100)
        return identity, tuple(
            Trajectory(slot, modes[slot], job.start, job.fps, tuple(samples))
            for slot, samples in tracks.items()
        )
    finally:
        frames.close()


class AnalysisWorker(QObject):
    progress = Signal(int)
    completed = Signal(object)
    failed = Signal(object)

    def __init__(self):
        super().__init__()
        self.cancel = threading.Event()

    @Slot(object)
    def run(self, job):
        try:
            identity, tracks = analyze_clip(job, self.cancel, self.progress.emit)
            self.completed.emit((job, identity, tracks))
        except AnalysisCancelled:
            self.failed.emit((job.token, "Анализ отменён"))
        except Exception as error:
            self.failed.emit((job.token, str(error)))
