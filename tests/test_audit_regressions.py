import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QLockFile, QObject, Signal
from PySide6.QtWidgets import QApplication

from framepilot.app import MainWindow
from framepilot.capture import WindowInfo
from framepilot.geometry import Rect, iou
from framepilot.preferences import Preferences, decode_selections
from framepilot.qt_capture import CapturePending, QtCapture, QtScreenInput
from framepilot.recovery import read_recovery
from framepilot.windows_camera_names import CameraNameError, Registration, remember_original


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def spin(app, predicate, timeout=8):
    until = time.monotonic() + timeout
    while not predicate() and time.monotonic() < until:
        app.processEvents()
        time.sleep(0.005)
    assert predicate()


def test_simultaneous_instances_do_not_share_settings_temporary_file(tmp_path):
    path = tmp_path / "preferences.json"
    errors = []
    gate = threading.Barrier(4)

    def write(index):
        gate.wait()
        for revision in range(30):
            try:
                Preferences(path).save({"instance": index, "revision": revision, "padding": "x" * 8192})
                assert len(json.loads(path.read_text())["padding"]) == 8192
            except Exception as error:
                errors.append(type(error).__name__)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(write, range(4)))
    assert errors == []


def test_close_during_processing_saves_newly_selected_photo_region(application, tmp_path):
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.full((100, 160, 3), (20, 60, 210), np.uint8))
    main = MainWindow()
    gate = threading.Event()
    timer = None
    try:
        main.load_photo(image)
        spin(application, lambda: main.photo_ready and not main.busy)
        process = main.video_worker.pipeline.process

        def delayed(*args, **kwargs):
            gate.wait(2)
            return process(*args, **kwargs)

        main.video_worker.pipeline.process = delayed
        subject = Rect(0.1, 0.2, 0.6, 0.5)
        main.select_object(subject)
        spin(application, lambda: main.busy)
        path = main.studio.recovery.path
        timer = threading.Timer(0.08, gate.set)
        timer.start()
        main.close()
        spin(application, lambda: not main.video_thread.isRunning())
        saved = read_recovery(path)
        assert saved is not None
        selection = decode_selections(saved["selections"])["single"]
        assert iou(selection, subject) > 0.999
    finally:
        gate.set()
        if timer is not None:
            timer.join()
        main.close()
        application.processEvents()


def test_windows_source_enumeration_error_is_reported_to_ui(application, monkeypatch):
    source = QtScreenInput()
    messages, listings = [], []
    source.message.connect(messages.append)
    source.sources.connect(listings.append)

    def fail():
        raise OSError("enumeration unavailable")

    monkeypatch.setattr(source.capture, "windows", fail)
    try:
        source.refresh()
        assert messages and "enumeration unavailable" in messages[-1]
        assert listings == [[]]
    finally:
        source.close()


@pytest.fixture
def fake_capture(application, monkeypatch):
    import framepilot.qt_capture as module

    class Session(QObject):
        def setWindowCapture(self, value):
            pass

        def setVideoSink(self, value):
            pass

    class Capture(QObject):
        errorOccurred = Signal(object, str)
        active = True

        def setWindow(self, value):
            pass

        def start(self):
            pass

        def stop(self):
            self.active = False

        def isActive(self):
            return self.active

    class Sink(QObject):
        videoFrameChanged = Signal(object)

    monkeypatch.setattr(module, "QMediaCaptureSession", Session)
    monkeypatch.setattr(module, "QWindowCapture", Capture)
    monkeypatch.setattr(module, "QVideoSink", Sink)
    capture = QtCapture(window_inspector=lambda identifier: (True, 54321, False))
    try:
        yield capture
    finally:
        capture.close()


def test_stopped_qt_capture_does_not_repeat_stale_image(fake_capture):
    capture = fake_capture
    window = WindowInfo(123, "External window", object())
    try:
        with pytest.raises(CapturePending):
            capture.capture(window)
        capture.image = np.ones((30, 40, 3), np.uint8)
        capture.capture_.active = False
        with pytest.raises(RuntimeError, match="останов"):
            capture.capture(window)
    finally:
        capture.close()


@pytest.mark.parametrize(
    "state, message",
    [((False, 0, False), "закрыто"), ((True, 54322, False), "изменилось"), ((True, 54321, True), "свёрнуто")],
)
def test_window_loss_cannot_be_reported_as_fresh_frames(fake_capture, state, message):
    capture = fake_capture
    capture.window_inspector = lambda identifier: (True, 54321, False)
    window = WindowInfo(123, "External", object(), process_id=54321)
    with pytest.raises(CapturePending):
        capture.capture(window)
    capture.image = np.ones((30, 40, 3), np.uint8)
    assert capture.capture(window).max() == 1
    capture.window_inspector = lambda identifier: state
    with pytest.raises(RuntimeError, match=message):
        capture.capture(window)
    assert capture.image is None


def test_static_windows_image_is_repeated_while_capture_remains_active(fake_capture):
    capture = fake_capture
    window = WindowInfo(123, "External", object())
    with pytest.raises(CapturePending):
        capture.capture(window)
    capture.image = np.ones((30, 40, 3), np.uint8)
    assert np.array_equal(capture.capture(window), capture.capture(window))


def test_window_capture_can_resume_after_source_error(application, monkeypatch):
    source = QtScreenInput()
    source.source = object()
    try:
        source.stop()
        source.toggle_pause()
        assert source.running and not source.paused and source.timer.isActive()
    finally:
        source.close()


def test_failed_obs_snapshot_does_not_change_obs(tmp_path, monkeypatch):
    import framepilot.app as gui
    import framepilot.connection_diagnostics as diagnostics
    import framepilot.connections as connections
    import framepilot.obs as obs

    config = tmp_path / "obs.json"
    config.write_text('{"server_enabled": true}')
    monkeypatch.setattr(obs, "websocket_config_path", lambda: config)
    mutations = []
    callbacks, exits = [], []

    class Timer:
        def __init__(self, *args):
            pass

        def stop(self):
            pass

        @staticmethod
        def singleShot(delay, callback):
            callbacks.append(callback)

    window = SimpleNamespace(obs_timer=Timer(), show=lambda: None, close=lambda: None)
    wizard = SimpleNamespace(
        receiver=SimpleNamespace(setCurrentIndex=lambda value: None, findData=lambda value: 0),
        show=lambda: None,
        reject=lambda: None,
        report=lambda: {},
    )
    monkeypatch.setattr(gui, "MainWindow", lambda: window)
    monkeypatch.setattr(connections, "ConnectionWizard", lambda parent: wizard)
    monkeypatch.setattr(diagnostics, "QTimer", Timer)

    def run_callbacks():
        while callbacks:
            callbacks.pop(0)()
        return exits[-1]

    application = SimpleNamespace(exec=run_callbacks, exit=exits.append)

    class Client:
        def __init__(self, *args):
            pass

        def connect(self):
            pass

        def close(self):
            pass

        def request(self, kind, data=None):
            if kind == "GetInputList":
                raise RuntimeError("snapshot interrupted")
            if not kind.startswith("Get"):
                mutations.append(kind)
            return {"outputActive": False, "currentProgramSceneName": "Original"}

    monkeypatch.setattr(obs, "OBSClient", Client)
    assert diagnostics.obs_self_test(application, tmp_path / "report.json") == 1
    assert mutations == []


def test_camera_original_backup_is_not_replaced_by_another_instance_lock(tmp_path):
    path = tmp_path / "original.json"
    lock = QLockFile(str(path) + ".lock")
    assert lock.tryLock(0)
    current = (Registration(64, r"C:\OBS\obs-virtualcam-module64.dll", "Initial", "Initial"),)
    try:
        with pytest.raises(CameraNameError):
            remember_original(path, current)
        assert not path.exists()
    finally:
        lock.unlock()


def test_cached_opaque_obs_device_id_survives_second_custom_name(monkeypatch):
    import framepilot.camera_identity as identity

    class Device:
        def id(self):
            return b"backend-opaque-id-unique"

        def description(self):
            return "First custom name"

    first = (
        Registration(64, r"C:\OBS\obs-virtualcam-module64.dll", "First custom name", "First custom name"),
    )
    identity.remember_obs_devices([Device()], first)
    renamed = (replace(first[0], friendly="Second custom name", class_name="Second custom name"),)
    assert identity.is_obs_camera(Device(), renamed)
    assert identity.camera_label(Device(), renamed) == "Second custom name"


def test_websocket_config_respects_windows_roaming_folder(tmp_path):
    from framepilot.obs import websocket_config_path

    windows = websocket_config_path(
        platform="win32", home=tmp_path / "user", environ={"APPDATA": str(tmp_path / "roaming")}
    )
    assert windows == tmp_path / "roaming/obs-studio/plugin_config/obs-websocket/config.json"
    mac = websocket_config_path(platform="darwin", home=tmp_path, environ={})
    assert mac == tmp_path / "Library/Application Support/obs-studio/plugin_config/obs-websocket/config.json"
