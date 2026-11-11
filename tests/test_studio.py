import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from framepilot.analysis import AnalysisCancelled, AnalysisJob, analyze_clip
from framepilot.app import MainWindow
from framepilot.appearance import install_appearance
from framepilot.automation import Automation, Cue, Sample, Trajectory, source_identity
from framepilot.exporter import ExportJob, export_video
from framepilot.geometry import Rect, iou
from framepilot.masks import FrameMask
from framepilot.pipeline import Pipeline, Renderer, Settings
from framepilot.recording import LiveRecorder
from framepilot.screen_source import FrameMailbox, content_sources, receiver_class
from framepilot.widgets import to_array


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    install_appearance(app)
    return app


@pytest.fixture
def window(application):
    window = MainWindow()
    window.show()
    application.processEvents()
    try:
        yield window
    finally:
        window.close()
        application.processEvents()
    assert not window.studio.analysis_thread.isRunning()
    assert not window.studio.screen.thread.isRunning()


def spin(app, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def moving_objects(tmp_path):
    path = tmp_path / "objects.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (240, 160))
    assert writer.isOpened()
    rng = np.random.default_rng(77)
    a = rng.integers(40, 250, (44, 44, 3), np.uint8)
    b = rng.integers(40, 250, (44, 44, 3), np.uint8)
    boxes = []
    for index in range(24):
        image = np.zeros((160, 240, 3), np.uint8)
        x = 15 + index
        image[30:74, x : x + 44] = a
        image[88:132, 160 - index : 204 - index] = b
        boxes.append(Rect(x / 240, 30 / 160, 44 / 240, 44 / 160))
        writer.write(image)
    writer.release()
    return path, boxes


def test_cue_interpolation_is_independent_per_slot_and_can_be_edited_at_current_time():
    a = Cue(1, "a", Rect(0.1, 0.1, 0.3, 0.4), Rect(0, 0, 0.5, 1), 0.2, 1)
    b = Cue(3, "a", Rect(0.5, 0.1, 0.3, 0.4), Rect(0, 0, 1, 1), 0.6, 2)
    automation = Automation((a, b))
    settings = Settings(automation_slot="a", automation=automation)
    middle = automation.settings_at(settings, 2)
    assert middle.padding == pytest.approx(0.4)
    assert middle.framing_zoom == pytest.approx(1.5)
    assert middle.subject_region == Rect(0, 0, 0.75, 1)
    assert automation.settings_at(replace(settings, automation_slot="other"), 2).subject_region == Rect(
        0, 0, 1, 1
    )
    assert automation.cue("a", 2) == a
    assert replace(automation, override_time=2, override_slot="a").settings_at(settings, 2) == settings
    assert Automation.from_record(automation.record()) == automation
    bad = automation.record()
    bad["cues"][0]["time"] = float("nan")
    with pytest.raises(ValueError):
        Automation.from_record(bad)


def test_cached_trajectory_interpolates_and_never_fills_a_lost_gap():
    track = Trajectory(
        "single",
        "object",
        0,
        15,
        (
            Sample(Rect(0.1, 0.1, 0.2, 0.2)),
            Sample(Rect(0.2, 0.1, 0.2, 0.2)),
            Sample(None, "lost"),
            Sample(None, "lost"),
        ),
    )
    assert track.at(1 / 30)[1].subject.x == pytest.approx(0.15)
    assert track.at(2 / 15)[1].subject is None
    assert not track.at(1)[0]
    assert track.problems() == [(2 / 15, 4 / 15, "lost")]
    pipeline = Pipeline()
    image = np.ones((80, 120, 3), np.uint8) * 200
    settings = Settings(
        width=120, height=80, subject_mode="object", loss="black", automation=Automation(tracks=(track,))
    )
    output, subject, _ = pipeline.process(image, 2 / 15, settings, None, True)
    assert subject is None and not output.any()
    _, subject, _ = pipeline.process(image, 0, settings, None, True)
    assert subject == track.samples[0].subject


@pytest.mark.parametrize("kind", ["rounded", "ellipse", "circle", "png"])
def test_masks_affect_real_composited_output_and_original_fallback(tmp_path, kind):
    mask_path = tmp_path / "alpha.png"
    mask = np.zeros((20, 20, 4), np.uint8)
    mask[2:18, 2:18, 3] = 128
    cv2.imwrite(str(mask_path), mask)
    settings = Settings(
        width=100,
        height=100,
        background_color="#00ff00",
        frame_mask=FrameMask(kind, 0.3, str(mask_path)),
        padding=0,
    )
    image = np.full((100, 100, 3), (255, 0, 0), np.uint8)
    renderer = Renderer()
    output = renderer.render(image, Rect(0, 0, 1, 1), settings, 0, snap=True)
    np.testing.assert_array_equal(output[0, 0], (0, 255, 0))
    if kind == "png":
        np.testing.assert_allclose(output[50, 50], (128, 127, 0), atol=1)
    else:
        np.testing.assert_array_equal(output[50, 50], (255, 0, 0))
    original = Renderer().render(image, None, replace(settings, loss="original"), 0)
    np.testing.assert_array_equal(original[0, 0], output[0, 0])
    np.testing.assert_array_equal(original[50, 50], output[50, 50])


def test_fit_keeps_wide_subject_inside_a_circular_mask():
    renderer = Renderer()
    settings = Settings(width=200, height=100, padding=0, frame_mask=FrameMask("circle"))
    renderer.render(np.ones((100, 100, 3), np.uint8) * 200, Rect(0, 0.25, 1, 0.5), settings, 0, snap=True)
    assert renderer.placement.scale * 100 <= 70.001


def test_screen_play_restarts_a_stopped_source_without_resetting_time(application, window, monkeypatch):
    screen = window.studio.screen
    screen.source = object()
    screen.running = screen.paused = False
    screen.epoch = 42
    started = []
    monkeypatch.setattr(screen, "start", lambda source, fps: started.append((source, fps)))
    screen.toggle_pause()
    assert started == [(screen.source, 30)]
    assert screen.epoch == 42


def test_screen_failure_and_stop_reject_late_processed_frames(application, window):
    window.active_source = "screen"
    for action, reason in (
        (lambda: window.studio.screen_message("Ошибка захвата"), "source_error"),
        (window.studio.stop_screen, "source_stopped"),
    ):
        session = window.session
        window.busy = True
        window.studio.screen.running = False
        window.studio.screen.paused = False
        action()
        assert window.session > session
        window.frame_ready(
            (session, np.full((720, 1280, 3), 255, np.uint8), Rect(0.1, 0.1, 0.2, 0.2), [], 1, "test")
        )
        assert window.publisher.hold_reason == reason
        assert not to_array(window.last_output).any()


def test_real_analysis_follows_selection_and_applies_later_retarget(moving_objects):
    path, boxes = moving_objects
    b_at_12 = Rect(148 / 240, 88 / 160, 44 / 240, 44 / 160)
    cue = Cue(12 / 15, "single", b_at_12, Rect(0, 0, 0.5, 1))
    settings = Settings(subject_mode="object", automation=Automation((cue,)))
    progress = []
    identity, tracks = analyze_clip(
        AnalysisJob(path, settings, boxes[0], 0, 15), threading.Event(), progress.append
    )
    assert identity == source_identity(path)
    assert len(tracks) == 1 and len(tracks[0].samples) == 24
    assert all(iou(tracks[0].samples[i].subject, boxes[i]) > 0.9 for i in range(12))
    assert iou(tracks[0].samples[12].subject, b_at_12) > 0.95
    assert all(s.subject.y > 0.5 for s in tracks[0].samples[12:])
    assert progress[-1] == 100


def test_cancel_analysis_does_not_return_partial_trajectory(moving_objects):
    path, boxes = moving_objects
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AnalysisCancelled):
        analyze_clip(AnalysisJob(path, Settings(subject_mode="object"), boxes[0], 0, 15), cancel)


def test_cached_analysis_and_mask_work_in_actual_mp4_export_without_reselecting(moving_objects, tmp_path):
    path, boxes = moving_objects
    settings = Settings(
        width=160, height=120, subject_mode="object", frame_mask=FrameMask("rounded", 0.3), padding=0
    )
    _, tracks = analyze_clip(AnalysisJob(path, settings, boxes[0], 0, 15), threading.Event())
    settings = replace(settings, automation=Automation(tracks=tracks))
    destination = tmp_path / "cached.mp4"
    assert export_video(ExportJob(path, destination, settings, 15, audio=False), threading.Event()) == 24
    frames = imageio_ffmpeg.read_frames(str(destination), pix_fmt="bgr24")
    metadata = next(frames)
    image = np.frombuffer(next(frames), np.uint8).reshape(120, 160, 3)
    frames.close()
    assert metadata["fps"] == 15
    assert image[0, 0].max() < 10 and image[60, 80].mean() > 30


def test_live_recording_uses_wall_clock_and_keeps_black_frames(tmp_path):
    recorder = LiveRecorder()
    path = tmp_path / "record.mp4"
    recorder.start(path, np.full((80, 120, 3), 230, np.uint8), 15)
    try:
        time.sleep(0.24)
        recorder.push(np.zeros((160, 90, 3), np.uint8))
        time.sleep(0.28)
    finally:
        recorder.close()
    assert recorder.error is None
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="bgr24")
    metadata = next(frames)
    images = [np.frombuffer(data, np.uint8) for data in frames]
    frames.close()
    assert metadata["size"] == (120, 80) and metadata["fps"] == 15
    assert len(images) >= 7 and len(images) == recorder.frames
    assert images[0].mean() > 220 and images[-1].max() < 8
    assert not list(tmp_path.glob(".framepilot-record-*"))


def test_screen_sources_exclude_own_windows_and_mailbox_is_bounded():
    own = SimpleNamespace(processID=lambda: 42, applicationName=lambda: "FramePilot")
    other = SimpleNamespace(processID=lambda: 99, applicationName=lambda: "Target")

    def window(app, id):
        return SimpleNamespace(
            owningApplication=lambda: app,
            windowLayer=lambda: 0,
            frame=lambda: SimpleNamespace(size=SimpleNamespace(width=400, height=300)),
            windowID=lambda: id,
            title=lambda: "Video",
        )

    display = SimpleNamespace(displayID=lambda: 1, width=lambda: 1920, height=lambda: 1080)
    content = SimpleNamespace(
        applications=lambda: [own, other],
        windows=lambda: [window(own, 1), window(other, 2)],
        displays=lambda: [display],
    )
    sources = content_sources(content, 42)
    assert len(sources) == 2 and sources[0].excluded_apps == (own,) and sources[1].identifier == 2
    mailbox = FrameMailbox()
    for index in range(100):
        mailbox.put(index, np.zeros((2, 2, 3), np.uint8))
    assert mailbox.take()[0] == 99
    assert mailbox.take() is None
    assert receiver_class().alloc().init() is not None


def test_analysis_project_roundtrip_and_stale_source_rejection(application, window, moving_objects, tmp_path):
    path, _ = moving_objects
    window.load_video(path)
    spin(application, lambda: window.latest_frame is not None and not window.busy)
    window.pause_source()
    window.studio.zoom.setValue(1.2)
    window.studio.mask_kind.setCurrentIndex(window.studio.mask_kind.findData("rounded"))
    window.studio.add_cue()
    # Switch to object selection to exercise the actual worker using a known textured region.
    frame_index = min(23, round(window.latest_media_time * 15))
    window.select_object(Rect((15 + frame_index) / 240, 30 / 160, 44 / 240, 44 / 160))
    spin(application, lambda: not window.busy and bool(window.input_view.boxes))
    window.studio.add_cue()
    window.studio.start_analysis()
    spin(application, lambda: not window.studio.analysis_busy)
    assert window.settings.automation.tracks, window.studio.analysis_status.text()
    original = window.settings.automation
    project = tmp_path / "clip.json"
    window.studio.save_project(project)
    window.studio.clear_analysis()
    window.studio.load_project(project)
    assert window.settings.automation == original
    assert window.settings.frame_mask.kind == "rounded"
    assert window.settings.framing_zoom == pytest.approx(1.2)
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="изменённому"):
        window.studio.load_project(project)
    assert window.settings.automation == original
    with pytest.raises(ValueError, match="Повторите анализ"):
        window.studio.validate_analysis_source()


def test_recording_encoder_failure_preserves_existing_destination(tmp_path, monkeypatch):
    path = tmp_path / "existing.mp4"
    path.write_bytes(b"original")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/usr/bin/false")
    recorder = LiveRecorder()
    recorder.start(path, np.ones((80, 120, 3), np.uint8), 15)
    recorder.close()
    assert recorder.error
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".framepilot-record-*"))


def test_mask_and_zoom_are_independent_per_scene_slot_and_survive_profiles(application, window):
    window.add_scene_slot()
    active = window.active_scene_slot
    window.studio.mask_kind.setCurrentIndex(window.studio.mask_kind.findData("circle"))
    window.studio.zoom.setValue(1.4)
    profile = window.capture_profile()
    window.select_scene_slot()
    other = next(s for s in window.scene_slots if s.identifier != active)
    window.scene_combo.setCurrentIndex(window.scene_combo.findData(other.identifier))
    assert window.studio.mask_kind.currentData() == "auto"
    assert window.studio.zoom.value() == 1
    window.apply_profile(profile)
    assert next(s for s in window.scene_slots if s.identifier == active).frame_mask.kind == "circle"
    assert next(s for s in window.scene_slots if s.identifier == active).framing_zoom == pytest.approx(1.4)


def test_record_button_pipeline_observes_emergency_black_and_never_overwrites_source(
    application, window, tmp_path
):
    output = np.full((window.settings.height, window.settings.width, 3), 190, np.uint8)
    window.show_processed(output)
    path = tmp_path / "live.mp4"
    window.studio.begin_recording(path)
    deadline = time.monotonic() + 0.18
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    window.toggle_black_output()
    assert not to_array(window.last_output).any()
    deadline = time.monotonic() + 0.18
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    window.studio.stop_recording()
    spin(application, lambda: not window.studio.recorder.active)
    assert window.studio.recorder.error is None
    window.video_path = path
    with pytest.raises(ValueError, match="исходный"):
        window.studio.begin_recording(path)
    video = imageio_ffmpeg.read_frames(str(path), pix_fmt="bgr24")
    next(video)
    images = [np.frombuffer(data, np.uint8) for data in video]
    video.close()
    assert images[0].mean() > 180 and images[-1].max() < 8
