"""Offline CFR MP4 export. All frames are processed; destination replaced only on success."""

import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from .automation import source_identity
from .geometry import Rect
from .object_geometry import QuadSelection
from .pipeline import Pipeline, Settings


@dataclass(frozen=True)
class ExportJob:
    source: Path
    destination: Path
    settings: Settings
    fps: int = 30
    start: float = 0
    audio: bool = True
    selection: Rect | tuple[float, float] | dict | None = None
    identity: dict | None = None
    frozen: bool = False
    queue_tag: str = ""


class ExportCancelled(Exception):
    pass


def export_video(job: ExportJob, cancel: threading.Event, progress=lambda _: None, pipeline_factory=Pipeline):
    job = replace(
        job, settings=replace(job.settings, automation=replace(job.settings.automation, override_time=None))
    )
    if job.source.resolve() == job.destination.resolve():
        raise ValueError("Нельзя заменить исходное видео результатом экспорта")
    if job.identity is not None and source_identity(job.source) != job.identity:
        raise ValueError("Исходный ролик изменился после подготовки задания")
    if job.settings.slots:
        selections = job.selection if isinstance(job.selection, dict) else {}
        for slot in job.settings.slots:
            cached, _ = job.settings.automation.cached(slot.identifier, slot.subject_mode, job.start)
            if (
                slot.subject_mode == "object"
                and not cached
                and not isinstance(selections.get(slot.identifier), (Rect, QuadSelection))
            ):
                raise ValueError("Подтвердите предмет в каждой рамке перед экспортом")
    elif (
        job.settings.subject_mode == "object"
        and not job.settings.automation.cached("single", "object", job.start)[0]
        and not isinstance(job.selection, (Rect, QuadSelection))
    ):
        raise ValueError("Сначала выделите предмет на входном видео")
    if job.fps not in (15, 24, 30) or job.start < 0:
        raise ValueError("Неверные параметры экспорта")
    job.destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".framepilot-", suffix=".mp4", dir=job.destination.parent)
    os.close(fd)
    temporary = Path(name)
    decoder = None
    encoder = None
    trimmed = None
    count = 0
    last_progress = -1
    try:
        if cancel.is_set():
            raise ExportCancelled()
        decoder = imageio_ffmpeg.read_frames(
            str(job.source),
            pix_fmt="bgr24",
            input_params=["-ss", f"{job.start:.6f}"] if job.start else [],
            output_params=["-vf", f"fps={job.fps}", "-threads", "1"],
        )
        metadata = next(decoder)
        width, height = metadata["size"]
        expected = max(1, (metadata.get("duration", 0) - job.start) * job.fps)
        pipeline = pipeline_factory()
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{job.settings.width}x{job.settings.height}",
            "-r",
            str(job.fps),
            "-i",
            "pipe:0",
        ]
        if job.audio:
            command += ["-ss", f"{job.start:.6f}", "-i", str(job.source), "-map", "0:v:0", "-map", "1:a:0?"]
        else:
            command += ["-map", "0:v:0", "-an"]
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
        ]
        if job.audio:
            command += ["-c:a", "aac", "-b:a", "192k", "-af", "apad"]
        command += ["-movflags", "+faststart", "-shortest", str(temporary)]
        with tempfile.TemporaryFile() as errors:
            encoder = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
            )
            try:
                for data in decoder:
                    if cancel.is_set():
                        raise ExportCancelled()
                    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                    output, _, _ = pipeline.process(
                        image,
                        job.start + count / job.fps,
                        job.settings,
                        job.selection if count == 0 else None,
                        count == 0,
                    )
                    encoder.stdin.write(output.tobytes())
                    count += 1
                    percentage = min(99, round(count / expected * 100))
                    if percentage != last_progress:
                        progress(percentage)
                        last_progress = percentage
            finally:
                if encoder.stdin:
                    encoder.stdin.close()
            if encoder.wait(timeout=30) != 0:
                errors.seek(0)
                raise RuntimeError("FFmpeg: " + errors.read().decode(errors="replace")[-1200:])
        if cancel.is_set():
            raise ExportCancelled()
        if count == 0:
            raise ValueError("В выбранном отрезке нет кадров")
        if job.audio:
            # Offline processing can stall input while FFmpeg has already buffered audio.
            # Remux to the exact emitted frame count instead of relying on -shortest queues.
            fd, trim_name = tempfile.mkstemp(
                prefix=".framepilot-audio-", suffix=".mp4", dir=job.destination.parent
            )
            os.close(fd)
            trimmed = Path(trim_name)
            with tempfile.TemporaryFile() as errors:
                encoder = subprocess.Popen(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(temporary),
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a:0?",
                        "-c",
                        "copy",
                        "-t",
                        f"{count / job.fps:.9f}",
                        "-movflags",
                        "+faststart",
                        str(trimmed),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                )
                while encoder.poll() is None:
                    if cancel.wait(0.1):
                        raise ExportCancelled()
                if encoder.returncode:
                    errors.seek(0)
                    raise RuntimeError("FFmpeg: " + errors.read().decode(errors="replace")[-1200:])
            os.replace(trimmed, temporary)
        if cancel.is_set():
            raise ExportCancelled()
        if job.identity is not None and source_identity(job.source) != job.identity:
            raise ValueError("Исходный ролик изменился во время экспорта")
        os.replace(temporary, job.destination)
        progress(100)
        return count
    finally:
        if decoder is not None:
            decoder.close()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            try:
                encoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
        temporary.unlink(missing_ok=True)
        if trimmed is not None:
            trimmed.unlink(missing_ok=True)
