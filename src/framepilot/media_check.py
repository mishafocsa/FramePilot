"""Bounded local first-frame decode check, independently of Qt's playback decoder."""

import re
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from PySide6.QtCore import QObject, QThread, Signal, Slot

from .automation import source_identity


def probe_video(path, cancelled=lambda: False):
    identity = source_identity(path)
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-probesize",
            "32M",
            "-analyzeduration",
            "5000000",
            "-i",
            str(path),
            "-map",
            "0:V:0",
            "-frames:v",
            "1",
            "-an",
            "-vf",
            "scale=64:64",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()
    try:
        while True:
            if cancelled():
                raise InterruptedError("Проверка отменена")
            if time.monotonic() - started > 15:
                raise ValueError("Проверка первого кадра превысила 15 секунд")
            try:
                data, log = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.returncode or len(data) != 64 * 64 * 3:
            raise ValueError(
                "FFmpeg не смог прочитать первый видеокадр. Файл может быть повреждён, защищён или содержать неподдерживаемый кодек."
            )
        text = log.decode(errors="replace")
        source = text.split("Output #", 1)[0]
        video = next((line for line in source.splitlines() if "Video:" in line), "")
        codec = re.search(r"Video:\s*([^ ,]+)", video)
        dimensions = re.search(r"\b(\d{2,6})x(\d{2,6})\b", video)
        fps = re.search(r"([\d.]+) fps", video)
        container = re.search(r"Input #0, (.+?), from", source)
        if source_identity(path) != identity:
            raise ValueError("Файл изменился во время проверки")
        return {
            "identity": identity,
            "codec": codec[1] if codec else "unknown",
            "size": [int(dimensions[1]), int(dimensions[2])] if dimensions else None,
            "fps": float(fps[1]) if fps else None,
            "container": container[1] if container else Path(path).suffix,
            "hdr": any(tag in source for tag in ("smpte2084", "arib-std-b67")),
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


class ProbeWorker(QObject):
    ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.latest = 0

    @Slot(object)
    def inspect(self, job):
        token, path = job
        if token != self.latest:
            return
        try:
            result = probe_video(path, lambda: token != self.latest)
            self.ready.emit((token, result, None))
        except InterruptedError:
            pass
        except Exception as error:
            self.ready.emit((token, None, str(error)))


class MediaCheck(QObject):
    request = Signal(object)
    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = QThread(self)
        self.worker = ProbeWorker()
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.inspect)
        self.worker.ready.connect(self.ready)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self.token = 0
        self.path = None
        self.result = None
        self.preview = False
        self.preview_error = None

    def start(self, path):
        self.token += 1
        self.worker.latest = self.token
        self.path = Path(path)
        self.result = None
        self.preview = False
        self.preview_error = None
        self.changed.emit("Проверяю кодек и первый кадр…")
        self.error = None
        self.request.emit((self.token, self.path))

    def ready(self, answer):
        token, result, error = answer
        if token != self.token:
            return
        self.result = result
        self.error = error
        self.refresh()

    def decoded(self):
        if not self.preview:
            self.preview = True
            self.refresh()

    def refresh(self):
        lines = []
        if self.result:
            r = self.result
            lines.append(
                f"{r['codec'].upper()} · {r['size'][0]}×{r['size'][1]}" if r["size"] else r["codec"].upper()
            )
            lines.append("Первый кадр FFmpeg прочитан")
            if r["hdr"]:
                lines.append("HDR-источник: точность преобразования цвета в SDR не проверена")
        elif getattr(self, "error", None):
            lines.append(self.error)
        else:
            lines.append("Проверка FFmpeg выполняется…")
        lines.append(
            "Предпросмотр Qt: кадр прочитан"
            if self.preview
            else "Предпросмотр Qt: " + (self.preview_error or "ожидание кадра")
        )
        self.changed.emit("\n".join(lines))

    def close(self):
        self.worker.latest = -1
        self.thread.quit()
        return self.thread.wait(17000)
