"""Asynchronous still-image input. Orientation and PNG transparency are preserved."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSize, QThread, Signal, Slot
from PySide6.QtGui import QColorSpace, QImage, QImageReader

from .automation import source_identity

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PhotoFrame:
    path: Path
    image: np.ndarray
    opacity: np.ndarray | None
    preview: QImage
    identity: dict
    original_size: tuple[int, int]


def read_photo(path):
    path = Path(path).resolve()
    identity = source_identity(path)
    reader = QImageReader(str(path))
    reader.setDecideFormatFromContent(True)
    reader.setAutoTransform(True)
    if not reader.canRead() or bytes(reader.format()).lower() not in (b"jpeg", b"jpg", b"png"):
        raise ValueError("Выберите исправное фото JPEG или PNG")
    size = reader.size()
    width, height = size.width(), size.height()
    if min(width, height) < 8 or width * height > 100_000_000:
        raise ValueError("Размер фото: от 8×8 до 100 миллионов пикселей")
    scale = min(1, 4096 / max(width, height))
    if scale < 1:
        reader.setScaledSize(QSize(max(1, round(width * scale)), max(1, round(height * scale))))
    image = reader.read()
    if image.isNull():
        raise ValueError("Не удалось прочитать фото: " + reader.errorString())
    if image.colorSpace().isValid():
        image = image.convertedToColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w, stride = image.height(), image.width(), image.bytesPerLine()
    rgba = (
        np.frombuffer(image.constBits(), np.uint8, count=h * stride)
        .reshape(h, stride)[:, : w * 4]
        .reshape(h, w, 4)
    )
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    opacity = rgba[:, :, 3].astype(np.float32) / 255 if np.any(rgba[:, :, 3] != 255) else None
    if source_identity(path) != identity:
        raise ValueError("Фото изменилось во время чтения")
    bgr.setflags(write=False)
    if opacity is not None:
        opacity.setflags(write=False)
    return PhotoFrame(path, bgr, opacity, image, identity, (width, height))


class PhotoWorker(QObject):
    loaded = Signal(object)
    failed = Signal(object)

    def __init__(self):
        super().__init__()
        self.latest = 0

    @Slot(object)
    def load(self, job):
        token, path = job
        if token != self.latest:
            return
        try:
            frame = read_photo(path)
            if token == self.latest:
                self.loaded.emit((token, frame))
        except Exception as error:
            if token == self.latest:
                self.failed.emit((token, str(error)))


class PhotoInput(QObject):
    request = Signal(object)
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.token = 0
        self.data = None
        self.thread = QThread(self)
        self.worker = PhotoWorker()
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.load)
        self.worker.loaded.connect(self.ready)
        self.worker.failed.connect(self.failure)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    def load(self, path):
        self.cancel()
        self.request.emit((self.token, Path(path)))

    def cancel(self):
        self.token += 1
        self.worker.latest = self.token
        self.data = None

    def ready(self, result):
        token, frame = result
        if token == self.token:
            self.data = frame
            self.loaded.emit(frame)

    def failure(self, result):
        if result[0] == self.token:
            self.failed.emit(result[1])

    def close(self):
        self.worker.latest = -1
        self.thread.quit()
        return self.thread.wait(17000)
