"""Windows window capture through Qt/FFmpeg. OS validation must run on Windows."""

import os
import sys
import time

import cv2
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtMultimedia import QMediaCaptureSession, QVideoSink, QWindowCapture

from .capture import WindowInfo
from .screen_source import ScreenSource
from .widgets import to_array


class CapturePending(RuntimeError):
    pass


def window_owner(identifier):
    import ctypes
    from ctypes import wintypes

    pid = wintypes.DWORD()
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowThreadProcessId(wintypes.HWND(identifier), ctypes.byref(pid))
    return pid.value


def window_state(identifier):
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    user.IsWindow.argtypes = user.IsIconic.argtypes = [wintypes.HWND]
    user.IsWindow.restype = user.IsIconic.restype = wintypes.BOOL
    handle = wintypes.HWND(identifier)
    return bool(user.IsWindow(handle)), window_owner(identifier), bool(user.IsIconic(handle))


class QtCapture(QObject):
    def __init__(self, parent=None, fps=15, window_inspector=None):
        super().__init__(parent)
        self.session = QMediaCaptureSession(self)
        self.capture_ = self.sink = None
        self.fps = fps
        self.generation = 0
        self.identifier = None
        self.image = None
        self.error = None
        self.window_inspector = window_inspector or (window_state if sys.platform == "win32" else None)
        self.owner = None

    def windows(self):
        result = []
        for window in QWindowCapture.capturableWindows():
            identifier = int(window.id())
            owner = window_owner(identifier)
            if not window.isValid() or owner in (0, os.getpid()):
                continue
            title = str(window.description())
            result.append(WindowInfo(identifier, title, window, "windows", title, owner))
        return sorted(result, key=lambda item: item.title.casefold())

    def received(self, frame):
        if frame.isValid():
            image = frame.toImage()
            if not image.isNull():
                array = to_array(image)
                h, w = array.shape[:2]
                scale = min(1, 1280 / max(w, h))
                self.image = (
                    cv2.resize(array, (max(1, round(w * scale)), max(1, round(h * scale))))
                    if scale < 1
                    else array
                )

    def failed(self, error, message):
        self.error = "Захват окна Windows: " + message
        self.image = None

    def capture(self, window):
        owner = None
        if self.window_inspector is not None:
            valid, owner, minimized = self.window_inspector(window.identifier)
            expected = window.process_id or (self.owner if window.identifier == self.identifier else None)
            if not valid or not owner or owner == os.getpid() or (expected and owner != expected):
                self.close()
                raise RuntimeError("Выбранное окно закрыто или изменилось. Выберите окно заново.")
            if minimized:
                self.close()
                raise RuntimeError("Выбранное окно свёрнуто. Разверните его и возобновите захват.")
        if window.identifier != self.identifier:
            self.close()
            self.identifier = window.identifier
            self.owner = owner
            generation = self.generation
            self.capture_ = QWindowCapture(self)
            self.sink = QVideoSink(self)
            self.session.setWindowCapture(self.capture_)
            self.session.setVideoSink(self.sink)
            if hasattr(self.capture_, "setMaximumFrameRate"):
                self.capture_.setMaximumFrameRate(float(self.fps))
            self.sink.videoFrameChanged.connect(
                lambda frame: self.received(frame) if generation == self.generation else None
            )
            self.capture_.errorOccurred.connect(
                lambda error, message: self.failed(error, message) if generation == self.generation else None
            )
            self.capture_.setWindow(window.native)
            self.capture_.start()
        if self.error:
            raise RuntimeError(self.error)
        if self.image is not None and hasattr(self.capture_, "isActive") and not self.capture_.isActive():
            self.close()
            raise RuntimeError("Захват окна остановился. Возобновите захват или выберите окно заново.")
        if self.image is None:
            raise CapturePending("Ожидание первого кадра окна Windows…")
        return self.image.copy()

    def close(self):
        self.generation += 1
        if self.capture_ is not None:
            self.capture_.stop()
            self.session.setWindowCapture(None)
            self.session.setVideoSink(None)
            self.capture_.deleteLater()
            self.sink.deleteLater()
            self.capture_ = self.sink = None
        self.identifier = None
        self.image = None
        self.error = None
        self.owner = None


class QtScreenInput(QObject):
    frame = Signal(object)
    sources = Signal(object)
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture = QtCapture(self, fps=30)
        self.source = None
        self.running = self.paused = False
        self.fps = 30
        self.epoch = time.monotonic()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.deliver)

    def refresh(self):
        try:
            sources = [
                ScreenSource("window", item.identifier, item.title, item.native, process_id=item.process_id)
                for item in self.capture.windows()
            ]
        except Exception as error:
            self.sources.emit([])
            self.message.emit("Получение списка окон Windows: " + str(error))
        else:
            self.sources.emit(sources)

    def start(self, source, fps):
        self.stop()
        self.source, self.fps = source, fps
        self.running = True
        self.epoch = time.monotonic()
        self.timer.start(round(1000 / fps))

    def deliver(self):
        if not self.running or self.source is None:
            return
        source = self.source
        try:
            image = self.capture.capture(
                WindowInfo(source.identifier, source.title, source.native, process_id=source.process_id)
            )
            # Qt can report no new callback for an unchanged window. Repeat the last frame.
            self.frame.emit((image, time.monotonic() - self.epoch))
        except CapturePending:
            pass
        except Exception as error:
            self.stop()
            self.message.emit(str(error))

    def stop(self):
        self.timer.stop()
        self.capture.close()
        self.running = self.paused = False

    def toggle_pause(self):
        if self.source is None:
            return
        if self.paused or not self.running:
            self.running, self.paused = True, False
            self.timer.start(round(1000 / self.fps))
        elif self.running:
            self.timer.stop()
            self.capture.close()
            self.running, self.paused = False, True

    def close(self):
        self.stop()
        return True
