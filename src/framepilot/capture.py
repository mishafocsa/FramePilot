"""macOS window capture. Invoked only from a capture thread after user selection."""

import os
import sys
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class WindowInfo:
    identifier: int
    title: str
    native: Any
    application: str = ""
    window_title: str = ""
    process_id: int | None = None

    @property
    def profile_key(self):
        return f"{self.application}::{self.window_title or self.title}"


class MacCapture:
    def __init__(self) -> None:
        import Quartz
        import ScreenCaptureKit

        self.q = Quartz
        self.sc = ScreenCaptureKit

    def windows(self) -> list[WindowInfo]:
        done = threading.Event()
        response = {}

        def complete(content, error):
            response.update(content=content, error=error)
            done.set()

        self.sc.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
            True, True, complete
        )
        if not done.wait(8):
            raise RuntimeError("macOS не ответила на запрос окон. Проверьте разрешение на запись экрана.")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        windows = []
        for window in response["content"].windows():
            app = window.owningApplication()
            if app is None or app.processID() == os.getpid() or window.windowLayer() != 0:
                continue
            size = window.frame().size
            if size.width < 150 or size.height < 120:
                continue
            title = f"{app.applicationName()} — {window.title() or 'Окно'}"
            windows.append(
                WindowInfo(
                    int(window.windowID()),
                    title,
                    window,
                    str(app.bundleIdentifier() or app.applicationName()),
                    str(window.title() or ""),
                )
            )
        return sorted(windows, key=lambda window: window.title)

    def capture(self, window: WindowInfo) -> np.ndarray:
        from Foundation import NSMutableData

        sc, q = self.sc, self.q
        filter_ = sc.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window.native)
        config = sc.SCStreamConfiguration.alloc().init()
        # Capture current content bounds, not stale window dimensions from enumeration.
        size = filter_.contentRect().size
        scale = min(1.0, 1000 / max(size.width, 1))
        config.setWidth_(max(2, int(size.width * scale)))
        config.setHeight_(max(2, int(size.height * scale)))
        config.setShowsCursor_(False)
        if hasattr(config, "setIgnoreShadowsSingleWindow_"):
            config.setIgnoreShadowsSingleWindow_(True)
        done = threading.Event()
        response = {}

        def complete(image, error):
            response.update(image=image, error=error)
            done.set()

        sc.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            filter_, config, complete
        )
        if not done.wait(5):
            raise RuntimeError("Таймаут захвата окна")
        if response.get("error") or response.get("image") is None:
            raise RuntimeError(str(response.get("error") or "Пустой снимок окна"))
        data = NSMutableData.data()
        destination = q.CGImageDestinationCreateWithData(data, "public.png", 1, None)
        q.CGImageDestinationAddImage(destination, response["image"], None)
        if not q.CGImageDestinationFinalize(destination):
            raise RuntimeError("Не удалось преобразовать снимок окна")
        decoded = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Не удалось прочитать снимок окна")
        return decoded


def create_capture():
    if sys.platform == "darwin":
        return MacCapture()
    if sys.platform == "win32":
        from .qt_capture import QtCapture

        return QtCapture()
    raise RuntimeError("Захват окон поддерживается в macOS и Windows")
