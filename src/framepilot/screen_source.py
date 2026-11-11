"""ScreenCaptureKit video input with own-app exclusion and a single-frame mailbox."""

import os
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot


@dataclass(frozen=True)
class ScreenSource:
    kind: str
    identifier: int
    title: str
    native: object
    excluded_apps: tuple = ()
    process_id: int | None = None


def content_sources(content, pid):
    own = tuple(app for app in content.applications() if app.processID() == pid)
    result = [
        ScreenSource("display", int(d.displayID()), f"Экран {i + 1} · {d.width()}×{d.height()}", d, own)
        for i, d in enumerate(content.displays())
    ]
    for window in content.windows():
        app = window.owningApplication()
        if app is None or app.processID() == pid or window.windowLayer() != 0:
            continue
        if window.frame().size.width < 100 or window.frame().size.height < 80:
            continue
        result.append(
            ScreenSource(
                "window",
                int(window.windowID()),
                f"{app.applicationName()} — {window.title() or 'Окно'}",
                window,
            )
        )
    return result


class FrameMailbox:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None

    def put(self, token, image):
        with self.lock:
            self.latest = token, image, time.monotonic()

    def take(self):
        with self.lock:
            result, self.latest = self.latest, None
        return result


@lru_cache(maxsize=1)
def receiver_class():
    import CoreMedia
    import objc
    import Quartz as q
    import ScreenCaptureKit as sc
    from Foundation import NSObject

    class FramePilotStreamReceiver(
        NSObject, protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")]
    ):
        def stream_didOutputSampleBuffer_ofType_(self, stream, sample, kind):
            try:
                if self.cancelled:
                    return
                if kind != sc.SCStreamOutputTypeScreen or not CoreMedia.CMSampleBufferIsValid(sample):
                    return
                attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sample, False)
                status = (
                    int(attachments[0].get(sc.SCStreamFrameInfoStatus, sc.SCFrameStatusComplete))
                    if attachments
                    else sc.SCFrameStatusComplete
                )
                if status == sc.SCFrameStatusIdle:
                    if self.last_image is not None:
                        self.mailbox.put(self.token, self.last_image)
                    return
                if status == sc.SCFrameStatusBlank:
                    if self.last_image is not None:
                        self.last_image = np.zeros_like(self.last_image)
                        self.mailbox.put(self.token, self.last_image)
                    return
                if status not in (sc.SCFrameStatusComplete, sc.SCFrameStatusStarted):
                    return
                pixel = CoreMedia.CMSampleBufferGetImageBuffer(sample)
                if pixel is None:
                    return
                flags = q.kCVPixelBufferLock_ReadOnly
                q.CVPixelBufferLockBaseAddress(pixel, flags)
                try:
                    width, height = q.CVPixelBufferGetWidth(pixel), q.CVPixelBufferGetHeight(pixel)
                    stride = q.CVPixelBufferGetBytesPerRow(pixel)
                    address = q.CVPixelBufferGetBaseAddress(pixel)
                    buffer = address.as_buffer(height * stride) if hasattr(address, "as_buffer") else address
                    bgra = (
                        np.frombuffer(buffer, np.uint8, count=height * stride)
                        .reshape(height, stride)[:, : width * 4]
                        .reshape(height, width, 4)
                    )
                    self.last_image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                finally:
                    q.CVPixelBufferUnlockBaseAddress(pixel, flags)
                self.mailbox.put(self.token, self.last_image)
            except Exception as error:
                if not self.cancelled:
                    self.error_signal.emit((self.token, "Захват экрана: " + str(error)))

        def stream_didStopWithError_(self, stream, error):
            if not self.cancelled:
                self.error_signal.emit((self.token, "Захват остановлен: " + str(error)))

    return FramePilotStreamReceiver


class ScreenWorker(QObject):
    listed = Signal(object)
    state = Signal(object)
    failed = Signal(object)

    def __init__(self, mailbox):
        super().__init__()
        self.mailbox = mailbox
        self.stream = self.receiver = None

    def content(self):
        import ScreenCaptureKit as sc

        done, response = threading.Event(), {}

        def complete(content, error):
            response.update(content=content, error=error)
            done.set()

        sc.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
            True, True, complete
        )
        if not done.wait(8) or response.get("error") or response.get("content") is None:
            raise RuntimeError(
                "Разрешите запись экрана для FramePilot в настройках macOS; после первого разрешения может потребоваться перезапуск."
            )
        return response["content"]

    @Slot()
    def list_sources(self):
        try:
            self.listed.emit(content_sources(self.content(), os.getpid()))
        except Exception as error:
            self.failed.emit((-1, str(error)))

    @Slot(object)
    def start(self, request):
        token, source, fps = request
        try:
            self.stop()
            import CoreMedia
            import Quartz as q
            import ScreenCaptureKit as sc

            if source.kind == "display":
                # Exclude every window owned by this process, including output and dialogs.
                if not source.excluded_apps:
                    raise ValueError("Не удалось определить собственное приложение для исключения из захвата")
                filter_ = sc.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
                    source.native, list(source.excluded_apps), []
                )
            else:
                if source.native.owningApplication().processID() == os.getpid():
                    raise ValueError("Собственное окно нельзя выбрать источником")
                filter_ = sc.SCContentFilter.alloc().initWithDesktopIndependentWindow_(source.native)
            rect = filter_.contentRect()
            scale = min(1, 960 / max(rect.size.width, rect.size.height, 1))
            config = sc.SCStreamConfiguration.alloc().init()
            config.setWidth_(max(2, round(rect.size.width * scale) // 2 * 2))
            config.setHeight_(max(2, round(rect.size.height * scale) // 2 * 2))
            config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, fps))
            config.setPixelFormat_(q.kCVPixelFormatType_32BGRA)
            config.setQueueDepth_(3)
            config.setShowsCursor_(False)
            if hasattr(config, "setPreservesAspectRatio_"):
                config.setPreservesAspectRatio_(True)
            if hasattr(config, "setColorSpaceName_"):
                config.setColorSpaceName_(q.kCGColorSpaceSRGB)
            if hasattr(config, "setIgnoreShadowsSingleWindow_"):
                config.setIgnoreShadowsSingleWindow_(True)
            receiver = receiver_class().alloc().init()
            receiver.cancelled = False
            receiver.mailbox, receiver.token, receiver.error_signal, receiver.last_image = (
                self.mailbox,
                token,
                self.failed,
                None,
            )
            stream = sc.SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, receiver)
            ok, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                receiver, sc.SCStreamOutputTypeScreen, None, None
            )
            if not ok:
                raise RuntimeError(str(error))
            self.stream, self.receiver = stream, receiver
            stream.startCaptureWithCompletionHandler_(
                lambda error: (
                    self.failed.emit((token, str(error))) if error else self.state.emit((token, "running"))
                )
            )
        except Exception as error:
            self.failed.emit((token, str(error)))

    @Slot()
    def stop(self):
        if self.stream is not None:
            self.receiver.cancelled = True
            done = threading.Event()
            stream, self.stream = self.stream, None
            stream.stopCaptureWithCompletionHandler_(lambda error: done.set())
            done.wait(5)
            self.receiver = None


class ScreenInput(QObject):
    frame = Signal(object)
    sources = Signal(object)
    message = Signal(str)
    list_request = Signal()
    start_request = Signal(object)
    stop_request = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mailbox = FrameMailbox()
        self.thread = QThread(self)
        self.worker = ScreenWorker(self.mailbox)
        self.worker.moveToThread(self.thread)
        self.list_request.connect(self.worker.list_sources)
        self.start_request.connect(self.worker.start)
        self.stop_request.connect(self.worker.stop)
        self.worker.listed.connect(self.sources.emit)
        self.worker.failed.connect(self.failed)
        self.worker.state.connect(self.state)
        self.thread.finished.connect(self.worker.stop)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self.token = 0
        self.running = self.paused = False
        self.source = None
        self.fps = 30
        self.epoch = time.monotonic()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.deliver)
        self.timer.start(10)

    def refresh(self):
        self.list_request.emit()

    def start(self, source, fps):
        self.stop()
        self.source, self.fps = source, fps
        self.running, self.paused = True, False
        self.epoch = time.monotonic()
        self.start_request.emit((self.token, source, fps))

    def stop(self):
        self.token += 1
        self.running = self.paused = False
        self.mailbox.take()
        self.stop_request.emit()

    def toggle_pause(self):
        if self.source is None:
            return
        if self.paused or not self.running:
            source, fps = self.source, self.fps
            epoch = self.epoch
            self.start(source, fps)
            self.epoch = epoch
        else:
            self.token += 1
            self.running, self.paused = False, True
            self.stop_request.emit()

    def deliver(self):
        frame = self.mailbox.take()
        if frame is not None and frame[0] == self.token and self.running and not self.paused:
            self.frame.emit((frame[1], frame[2] - self.epoch))

    def failed(self, result):
        token, message = result
        if token in (-1, self.token):
            if token != -1:
                self.running = False
            self.message.emit(message)

    def state(self, result):
        if result[0] == self.token:
            self.message.emit("Захват экрана включён")

    def close(self):
        self.timer.stop()
        self.stop()
        self.thread.quit()
        return self.thread.wait(14000)


def create_screen_input(parent=None):
    if sys.platform == "win32":
        from .qt_capture import QtScreenInput

        return QtScreenInput(parent)
    return ScreenInput(parent)
