"""Bounded, wall-clock CFR recording of the displayed output; video only."""

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


class LiveRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.latest = None
        self.frames = 0
        self.error = None
        self.destination = None
        self.started = 0
        self.size = None
        self.fps = 30

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, destination, initial, fps):
        if self.active:
            raise ValueError("Запись уже идёт")
        if fps not in (15, 24, 30) or initial is None or initial.ndim != 3 or initial.shape[2] != 3:
            raise ValueError("Нет готового изображения для записи")
        self.size = initial.shape[1], initial.shape[0]
        if min(self.size) < 2 or any(v % 2 for v in self.size):
            raise ValueError("Для MP4 нужны чётные размеры кадра")
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".framepilot-record-", suffix=".mp4", dir=self.destination.parent)
        os.close(fd)
        self.temporary = Path(name)
        self.process = None
        self.fps, self.latest = fps, initial.copy()
        self.frames, self.error = 0, None
        self.stop_event.clear()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, name="FramePilot recording", daemon=True)
        self.thread.start()

    def push(self, image):
        if self.active:
            with self.lock:
                self.latest = image.copy()

    def stop(self):
        self.stop_event.set()

    def close(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(5)
            if self.thread.is_alive() and self.process is not None:
                self.process.kill()
                self.thread.join(3)

    def fitted(self, image):
        width, height = self.size
        if image.shape[:2] == (height, width):
            return image
        h, w = image.shape[:2]
        scale = min(width / w, height / h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        result = np.zeros((height, width, 3), np.uint8)
        x, y = (width - rw) // 2, (height - rh) // 2
        result[y : y + rh, x : x + rw] = cv2.resize(image, (rw, rh))
        return result

    def _run(self):
        temporary = self.temporary
        width, height = self.size
        try:
            with tempfile.TemporaryFile() as errors:
                self.process = subprocess.Popen(
                    [
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
                        f"{width}x{height}",
                        "-r",
                        str(self.fps),
                        "-i",
                        "pipe:0",
                        "-an",
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
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                )
                try:
                    while True:
                        due = self.started + self.frames / self.fps
                        delay = due - time.monotonic()
                        if self.frames and self.stop_event.wait(max(0, delay)):
                            break
                        if time.monotonic() - due > 2:
                            raise RuntimeError(
                                "Кодировщик отстаёт больше чем на 2 секунды; запись остановлена"
                            )
                        with self.lock:
                            image = self.latest
                        self.process.stdin.write(self.fitted(image).tobytes())
                        self.frames += 1
                        if self.stop_event.is_set():
                            break
                finally:
                    self.process.stdin.close()
                if self.process.wait(timeout=15):
                    errors.seek(0)
                    raise RuntimeError("FFmpeg: " + errors.read().decode(errors="replace")[-1000:])
            if not self.frames:
                raise ValueError("В записи нет кадров")
            os.replace(temporary, self.destination)
        except Exception as error:
            self.error = str(error)
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=3)
            temporary.unlink(missing_ok=True)
