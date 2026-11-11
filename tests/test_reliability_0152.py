import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication

from framepilot.app import MainWindow
from framepilot.automation import Automation, Sample, Trajectory
from framepilot.batch import BatchExport
from framepilot.exporter import ExportJob
from framepilot.geometry import Rect
from framepilot.photos import read_photo
from framepilot.pipeline import Pipeline, Settings
from framepilot.studio import Studio


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("collision", ["source", "unrelated_file"])
def test_project_save_preserves_existing_file_at_old_temporary_name(tmp_path, collision):
    destination = tmp_path / "clip.framepilot.json"
    old_temporary = destination.with_suffix(destination.suffix + ".tmp")
    old_temporary.write_bytes(b"irreplaceable original bytes")
    source = old_temporary if collision == "source" else tmp_path / "input.avi"
    if source != old_temporary:
        source.write_bytes(b"video original")
    main = SimpleNamespace(
        active_source="file",
        video_path=source,
        settings=Settings(automation=Automation()),
        capture_profile=lambda: {},
    )
    studio = SimpleNamespace(main=main, analysis_identity=None)
    Studio.save_project(studio, destination)
    assert old_temporary.read_bytes() == b"irreplaceable original bytes"
    assert json.loads(destination.read_text())["format"] == "framepilot-clip"


def test_maximum_trajectory_stays_within_project_file_size_limit(tmp_path):
    source = tmp_path / "input.avi"
    source.write_bytes(b"unchanged source")
    sample = Sample(
        Rect(0.12345678912345678, 0.23456789123456789, 0.34567891234567891, 0.45678912345678912),
        eyes=(0.23456789123456789, 0.34567891234567891),
        quad=(
            (0.12345678912345678, 0.23456789123456789),
            (0.46913570146913569, 0.23456789123456789),
            (0.46913570146913569, 0.69135701469135701),
            (0.12345678912345678, 0.69135701469135701),
        ),
    )
    automation = Automation(tracks=(Trajectory("single", "object", 0, 30, (sample,) * 200000),))
    main = SimpleNamespace(
        active_source="file",
        video_path=source,
        settings=Settings(automation=automation),
        capture_profile=lambda: {},
    )
    studio = SimpleNamespace(main=main, analysis_identity=None)
    destination = tmp_path / "large.framepilot.json"
    Studio.save_project(studio, destination)
    assert destination.stat().st_size <= 64 * 1024 * 1024


@pytest.mark.parametrize("order", ["result_overwrites_input", "input_is_previous_result"])
def test_queue_cannot_overwrite_another_jobs_original(tmp_path, order):
    a, b, result = (tmp_path / name for name in ("a.avi", "b.avi", "output.mp4"))
    a.write_bytes(b"original a")
    b.write_bytes(b"original b")
    queue = SimpleNamespace(entries=[], refresh=lambda: None)
    settings = Settings()
    if order == "result_overwrites_input":
        first, second = ExportJob(a, result, settings), ExportJob(b, a, settings)
    else:
        first, second = ExportJob(a, b, settings), ExportJob(b, result, settings)
    BatchExport.add(queue, first)
    with pytest.raises(ValueError, match="исходн"):
        BatchExport.add(queue, second)
    assert len(queue.entries) == 1
    assert a.read_bytes() == b"original a" and b.read_bytes() == b"original b"


@pytest.fixture
def camera(application, monkeypatch):
    import framepilot.camera as module

    class Session(QObject):
        def setVideoSink(self, sink):
            self.sink = sink

        def setCamera(self, camera):
            self.camera = camera

    class Sink(QObject):
        videoFrameChanged = Signal(object)

    class Device(QObject):
        errorOccurred = Signal(object, str)

        def __init__(self, device, parent):
            super().__init__(parent)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(module, "QMediaCaptureSession", Session)
    monkeypatch.setattr(module, "QVideoSink", Sink)
    monkeypatch.setattr(module, "QCamera", Device)
    monkeypatch.setattr(application, "checkPermission", lambda _: Qt.PermissionStatus.Granted)
    controller = module.CameraInput()
    try:
        yield controller
    finally:
        controller.stop()
        application.processEvents()


@pytest.mark.parametrize("action", ["stop", "switch"])
def test_late_camera_error_does_not_fail_a_new_source(camera, action):
    camera.start(object())
    old_camera = camera.camera
    if action == "switch":
        camera.start(object())
    else:
        camera.stop()
    messages = []
    camera.message.connect(messages.append)
    old_camera.errorOccurred.emit(object(), "old device disconnected")
    assert messages == []


def test_late_frame_from_previous_camera_is_discarded(camera):
    camera.start(object())
    old_sink = camera.sink
    camera.start(object())
    frames = []
    camera.frame.connect(frames.append)
    old_sink.videoFrameChanged.emit("old frame")
    assert frames == []
    camera.sink.videoFrameChanged.emit("new frame")
    assert frames == ["new frame"]


def test_cancelled_permission_denial_does_not_fail_current_camera(camera, application, monkeypatch):
    callbacks = []
    monkeypatch.setattr(application, "checkPermission", lambda _: Qt.PermissionStatus.Undetermined)
    monkeypatch.setattr(application, "requestPermission", lambda p, ctx, cb: callbacks.append(cb))
    camera.start(object())
    monkeypatch.setattr(application, "checkPermission", lambda _: Qt.PermissionStatus.Granted)
    camera.start(object())
    messages = []
    camera.message.connect(messages.append)
    callbacks[0](SimpleNamespace(status=lambda: Qt.PermissionStatus.Denied))
    assert messages == [] and camera.camera is not None


def test_very_wide_photo_preserves_its_aspect_ratio(application, tmp_path):
    path = tmp_path / "panorama.png"
    assert cv2.imwrite(str(path), np.full((16, 32768, 3), 130, np.uint8))
    photo = read_photo(path)
    assert photo.image.shape[:2] == (2, 4096)


def test_very_wide_source_never_resizes_to_zero_pixels(monkeypatch):
    monkeypatch.setenv("FRAMEPILOT_VISION_BACKEND", "yunet")
    image = np.full((2, 4096, 3), 130, np.uint8)
    pipeline = Pipeline()
    settings = Settings(width=320, height=240, subject_mode="object")
    output, subject, _ = pipeline.process(image, 0, settings, Rect(0, 0, 1, 1), True, still=True)
    assert output.shape == (240, 320, 3)
    assert subject == Rect(0, 0, 1, 1)


@pytest.mark.parametrize("source", ["photo", "camera", "screen"])
def test_late_video_end_cannot_reset_selection_in_other_sources(source):
    restarts = []
    main = SimpleNamespace(active_source=source, restart=lambda: restarts.append(True))
    MainWindow.media_status(main, QMediaPlayer.MediaStatus.EndOfMedia)
    assert restarts == []


@pytest.mark.parametrize("paused, stale, expected", [(True, False, 0), (False, True, 0), (False, False, 1)])
def test_video_loop_uses_current_media_state(paused, stale, expected):
    restarts = []
    main = SimpleNamespace(
        active_source="file",
        closing_requested=False,
        loop=SimpleNamespace(isChecked=lambda: True),
        player=SimpleNamespace(
            playbackState=lambda: (
                QMediaPlayer.PlaybackState.PausedState if paused else QMediaPlayer.PlaybackState.StoppedState
            ),
            mediaStatus=lambda: (
                QMediaPlayer.MediaStatus.LoadedMedia if stale else QMediaPlayer.MediaStatus.EndOfMedia
            ),
        ),
        restart=lambda: restarts.append(True),
    )
    MainWindow.media_status(main, QMediaPlayer.MediaStatus.EndOfMedia)
    assert len(restarts) == expected


@pytest.mark.parametrize("source", ["photo", "camera", "screen"])
def test_old_video_error_does_not_replace_current_source_status(source):
    messages = []
    main = SimpleNamespace(
        active_source=source, closing_requested=False, status=SimpleNamespace(setText=messages.append)
    )
    MainWindow.player_error(main, QMediaPlayer.Error.ResourceError, "old video failed")
    assert messages == []
