import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from framepilot.app import MainWindow
from framepilot.capture import WindowInfo
from framepilot.debug_bundle import save_debug_bundle
from framepilot.geometry import Rect
from framepilot.qt_capture import CapturePending, QtCapture, QtScreenInput
from framepilot.widgets import to_image


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_old_window_frame_cannot_cross_into_new_window(application, monkeypatch):
    import framepilot.qt_capture as module

    class Session(QObject):
        def setWindowCapture(self, capture):
            self.capture = capture

        def setVideoSink(self, sink):
            self.sink = sink

    class Capture(QObject):
        errorOccurred = Signal(object, str)

        def setWindow(self, window):
            self.window = window

        def start(self):
            pass

        def stop(self):
            pass

    class Sink(QObject):
        videoFrameChanged = Signal(object)

    class Frame:
        def isValid(self):
            return True

        def toImage(self):
            return to_image(np.full((30, 40, 3), 210, np.uint8))

    monkeypatch.setattr(module, "QMediaCaptureSession", Session)
    monkeypatch.setattr(module, "QWindowCapture", Capture)
    monkeypatch.setattr(module, "QVideoSink", Sink)
    capture = QtCapture(window_inspector=lambda identifier: (True, 54321, False))
    first, second = WindowInfo(1, "A", object()), WindowInfo(2, "B", object())
    with pytest.raises(CapturePending):
        capture.capture(first)
    old_sink = capture.sink
    old_sink.videoFrameChanged.emit(Frame())
    assert capture.capture(first).max() == 210
    with pytest.raises(CapturePending):
        capture.capture(second)
    old_sink.videoFrameChanged.emit(Frame())
    with pytest.raises(CapturePending):
        capture.capture(second)
    capture.sink.videoFrameChanged.emit(Frame())
    assert capture.capture(second).shape == (30, 40, 3)
    capture.close()
    application.processEvents()


def test_windows_ui_exposes_portable_modes_without_loading_apple_apis(application, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    window = MainWindow()
    try:
        assert isinstance(window.studio.screen, QtScreenInput)
        assert not window.remove_background.isEnabled()
        assert not window.subject_mode.model().item(2).isEnabled()
        profile = window.capture_profile()
        profile["subject_mode"] = "person"
        profile["remove_background"] = True
        window.apply_profile(profile)
        assert window.settings.subject_mode == "face"
        assert not window.settings.remove_background
    finally:
        window.close()
        application.processEvents()


def test_error_bundle_contains_visible_frames_and_selection_without_connection_secrets(application, tmp_path):
    window = MainWindow()
    try:
        window.input_view.show_image(to_image(np.full((50, 80, 3), 100, np.uint8)))
        window.target_view.show_image(to_image(np.full((50, 80, 3), 140, np.uint8)))
        window.show_processed(np.full((50, 80, 3), 180, np.uint8))
        window.last_tracks = {"single": {"subject": Rect(0.1, 0.1, 0.4, 0.6)}}
        window.connection_profile = {"password": "never-save-this"}
        folder = save_debug_bundle(window, tmp_path)
        text = (folder / "case.json").read_text()
        data = json.loads(text)
        assert "never-save-this" not in text
        assert data["tracking"]["single"]["subject"]["w"] == 0.4
        assert all((folder / (name + ".png")).exists() for name in ("input", "target", "output", "interface"))
    finally:
        window.close()
        application.processEvents()
