import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication

import framepilot
from framepilot.app import STYLE, MainWindow
from framepilot.calibration import Calibration
from framepilot.geometry import Rect, iou
from framepilot.vision import detect_slots
from framepilot.widgets import ImageCanvas, to_array, to_image


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    yield app


def spin(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_image_round_trip_handles_padded_rows(application):
    array = np.random.default_rng(42).integers(0, 255, size=(29, 101, 3), dtype=np.uint8)
    np.testing.assert_array_equal(to_array(to_image(array)), array)


def test_click_mapping_accounts_for_letterboxing(application):
    canvas = ImageCanvas()
    canvas.resize(400, 400)
    canvas.show_image(to_image(np.zeros((100, 200, 3), np.uint8)))
    assert canvas.normalized(QPointF(200, 200)) == (0.5, 0.5)
    assert canvas.normalized(QPointF(200, 30)) is None


def test_demo_is_detected_from_actual_widget_pixels_and_workers_stop(application):
    window = MainWindow()
    try:
        window.demo.timer.stop()
        window.demo.show()
        application.processEvents()
        for index in range(4):
            window.demo.index = index
            window.demo.repaint()
            application.processEvents()
            image = to_array(window.demo.grab().toImage())
            slots = detect_slots(image)
            expected, shape = window.demo.slot()
            assert slots and slots[0].shape == shape
            assert iou(slots[0].rect, expected) > 0.9
        window.scan_window()
        spin(application, lambda: not window.scan_busy)
        window.scan_window()
        spin(application, lambda: not window.scan_busy)
        assert window.stable.accepted is not None
        assert window.settings.shape == "ellipse"
        # Manual selection invalidates an already running automatic result.
        old_generation = window.generation
        manual = Rect(0.1, 0.1, 0.3, 0.4)
        window.manual_frame(manual)
        window.scan_ready((old_generation, image, slots))
        assert window.settings.target == manual
    finally:
        window.close()
        application.processEvents()
    assert not window.video_thread.isRunning()
    assert not window.capture_thread.isRunning()


def test_manual_target_continues_capturing_without_overwriting_frame(application):
    window = MainWindow()
    try:
        window.scan_timer.stop()
        window.demo.show()
        window.auto_slot.setChecked(False)
        frame = Rect(0.2, 0.2, 0.4, 0.5)
        window.manual_frame(frame)
        window.target_view.show_image(QImage())
        window.scan_window()
        spin(application, lambda: not window.scan_busy)
        assert not window.target_view.image.isNull()
        assert window.settings.target == frame
    finally:
        window.close()


def test_actual_video_playback_reaches_face_tracking_and_output(application, tmp_path):
    portrait = cv2.imread(str(Path(framepilot.__file__).parent / "assets" / "astronaut.png"))
    path = tmp_path / "moving-face.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (512, 512))
    assert writer.isOpened()
    for index in range(72):
        shifted = cv2.warpAffine(portrait, np.float32([[1, 0, index % 30], [0, 1, 0]]), (512, 512))
        writer.write(shifted)
    writer.release()
    window = MainWindow()
    try:
        window.show()
        window.show_demo()
        window.demo.animate = False
        window.load_video(path)
        spin(
            application, lambda: not window.output_view.image.isNull() or "Ошибка" in window.status.text(), 20
        )
        assert not window.output_view.image.isNull(), window.status.text()
        spin(application, lambda: "Лицо отслеживается" in window.status.text(), 20)
        spin(application, lambda: window.stable.accepted is not None, 10)
        spin(
            application,
            lambda: (
                window.output_view.image.width() == window.settings.width
                and window.output_view.image.height() == window.settings.height
            ),
            10,
        )
        assert window.input_view.boxes
        window.player.pause()
        spin(application, lambda: not window.busy)
        window.resolution.setCurrentIndex(2)
        spin(
            application,
            lambda: (
                not window.busy
                and not window.render_pending
                and window.output_view.image.width() == 720
                and window.output_view.image.height() == 1280
            ),
            10,
        )
        assert window.input_view.boxes, (
            window.status.text(),
            window.video_worker.pipeline.last_face,
            window.video_worker.pipeline.tracker.box,
            window.video_worker.pipeline.tracker.last_seen,
            window.latest_media_time,
            window.reset_pending,
            window.video_worker.pipeline.tracker.count,
        )
        assert window.status.text().startswith("Пауза")
        window.seek_to(1500)
        spin(application, lambda: not window.busy and window.latest_frame is not None, 10)
        assert window.timeline.maximum() > 0
        assert abs(window.player.position() - 1500) < 100
        window.show_output()
        application.processEvents()
        screenshot = Path("test-results")
        screenshot.mkdir(exist_ok=True)
        window.grab().save(str(screenshot / "interface.png"))
        window.output_window.grab().save(str(screenshot / "output.png"))
        window.demo.grab().save(str(screenshot / "receiving-window.png"))
    finally:
        window.close()
        application.processEvents()


def test_preferences_roundtrip_and_corrupt_profile_does_not_partially_apply(application):
    window = MainWindow()
    try:
        window.padding.setValue(0.55)
        window.sensitivity.setCurrentIndex(1)
        window.preferred_port = 18888
        profile = window.capture_profile()
        profile["padding"] = 0.9
        profile["fps"] = 99
        with pytest.raises(ValueError):
            window.apply_profile(profile)
        assert window.padding.value() == 0.55
    finally:
        window.close()
    restored = MainWindow()
    try:
        assert restored.padding.value() == 0.55
        assert restored.sensitivity.currentData() == "sensitive"
        assert restored.preferred_port == 18888
    finally:
        restored.close()


def test_late_video_frame_does_not_change_paused_preview(application, monkeypatch):
    window = MainWindow()

    class LateFrame:
        def isValid(self):
            return True

        def toImage(self):
            return to_image(np.zeros((100, 100, 3), np.uint8))

        def startTime(self):
            return 100_000

    try:
        monkeypatch.setattr(window.player, "playbackState", lambda: QMediaPlayer.PlaybackState.PausedState)
        window.video_frame(LateFrame())
        assert not window.busy
        assert window.latest_frame is None
    finally:
        window.close()


def test_object_selection_grid_and_named_window_profile(application):
    window = MainWindow()
    try:
        image = np.zeros((180, 260, 3), np.uint8)
        image[60:110, 40:100] = np.random.default_rng(5).integers(40, 255, (50, 60, 3), np.uint8)
        window.latest_frame = image
        window.input_view.show_image(to_image(image))
        window.select_object(Rect(40 / 260, 60 / 180, 60 / 260, 50 / 180))
        spin(application, lambda: not window.busy and "Предмет отслеживается" in window.status.text())
        assert window.object_seed is not None
        assert window.settings.subject_mode == "object"
        window.settings = replace(window.settings, calibration=Calibration(Rect(0.1, 0.1, 0.8, 0.8), True))
        window.read_settings()
        window.remember_window_profile()
        window.padding.setValue(0.7)
        window.apply_named_profile("demo")
        assert window.padding.value() == 0.25
        assert window.settings.calibration.mirror
        window.grid.setChecked(True)
        assert not window.last_output.isNull()
        raw_width = window.last_output.width()
        window.receiver_preview.setChecked(True)
        assert window.output_view.image.width() < raw_width
        assert window.output_window.image.width() == raw_width
        window.grid.setChecked(False)
        spin(application, lambda: not window.busy)
    finally:
        window.close()


def test_obs_debounce_discards_intermediate_resolution(application):
    window = MainWindow()
    try:
        window.preferred_port = 0
        window.start_local_output()
        window.obs_connected = True
        window.obs_last_synced = (window.publisher.url, 1280, 720, 30)
        window.resolution.setCurrentIndex(2)
        assert window.obs_pending[1:3] == (720, 1280)
        window.resolution.setCurrentIndex(0)
        assert window.obs_pending is None
        assert not window.obs_sync_timer.isActive()
    finally:
        window.close()


def test_multiple_slots_keep_independent_targets_and_profile_settings(application):
    window = MainWindow()
    try:
        window.add_scene_slot()
        assert len(window.scene_slots) == 2
        window.subject_mode.setCurrentIndex(window.subject_mode.findData("object"))
        window.padding.setValue(0.65)
        right_id = window.active_scene_slot
        rectangle = Rect(0.6, 0.3, 0.25, 0.6)
        window.manual_frame(rectangle)
        window.scene_combo.setCurrentIndex(0)
        assert window.padding.value() == 0.25
        assert window.settings.subject_mode == "face"
        profile = window.capture_profile()
        saved = next(slot for slot in profile["slots"] if slot["id"] == right_id)
        assert saved["padding"] == 0.65
        assert saved["target"] == [0.6, 0.3, 0.25, 0.6]
        window.apply_profile(profile)
        assert len(window.settings.slots) == 2
        window.scene_combo.setCurrentIndex(1)
        assert window.padding.value() == 0.65
        assert window.settings.target == rectangle
        assert window.settings.subject_mode == "object"
        broken = dict(profile, slots=profile["slots"] * 2)
        with pytest.raises(ValueError, match="уникальными"):
            window.apply_profile(broken)
        assert window.settings.target == rectangle
        sequences = {shortcut.key().toString() for shortcut in window.findChildren(QShortcut)}
        assert QKeySequence("Ctrl+Shift+B").toString() in sequences
    finally:
        window.close()


def test_scenario_restores_source_time_targets_and_layout(application, tmp_path):
    window = MainWindow()
    try:
        window.load_sample()
        spin(application, lambda: bool(window.last_tracks) and not window.busy, 20)
        window.pause_source()
        window.add_scene_slot()
        window.remove_background.setChecked(True)
        spin(application, lambda: len(window.last_tracks) == 2 and not window.busy, 20)
        window.save_scenario("Два плана")
        entry = next(iter(window.scenarios.values()))
        assert entry["selections"]
        window.remove_scene_slot()
        window.remove_background.setChecked(False)
        window.apply_scenario()
        spin(
            application,
            lambda: window.pending_scenario is None and len(window.last_tracks) == 2 and not window.busy,
            20,
        )
        window.pause_source()
        assert window.remove_background.isChecked()
        assert len(window.settings.slots) == 2
        assert window.video_path == Path(entry["path"])
        restored = window.preferences.load()["scenarios"]
        assert next(iter(restored.values()))["name"] == "Два плана"
        # Missing media must fail before any of the current settings are mutated.
        entry["path"] = str(tmp_path / "missing.mp4")
        window.apply_scenario()
        assert "не найден" in window.status.text()
        assert len(window.settings.slots) == 2
    finally:
        window.close()


def test_camera_permission_and_late_grant_cancellation_without_opening_hardware(application, monkeypatch):
    import framepilot.camera as camera_module

    opened = []
    pending = []

    class FakeCamera(QObject):
        errorOccurred = Signal(object, str)

        def __init__(self, device, parent):
            super().__init__(parent)
            self.running = False
            opened.append(self)

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    controller = camera_module.CameraInput()
    monkeypatch.setattr(camera_module, "QCamera", FakeCamera)
    monkeypatch.setattr(controller.session, "setCamera", lambda camera: None)
    monkeypatch.setattr(application, "checkPermission", lambda permission: Qt.PermissionStatus.Undetermined)
    monkeypatch.setattr(
        application, "requestPermission", lambda permission, context, callback: pending.append(callback)
    )
    controller.start(object())
    controller.stop()

    class Granted:
        def status(self):
            return Qt.PermissionStatus.Granted

    pending[0](Granted())
    assert not opened
    monkeypatch.setattr(application, "checkPermission", lambda permission: Qt.PermissionStatus.Granted)
    controller.start(object())
    assert len(opened) == 1 and opened[0].running
    controller.toggle_pause()
    assert controller.paused and not opened[0].running
    controller.stop()
    assert controller.camera is None and not opened[0].running


def test_camera_source_discards_late_file_frames_and_paused_live_frames(application, monkeypatch):
    window = MainWindow()

    class Frame:
        def isValid(self):
            return True

        def toImage(self):
            return to_image(np.full((100, 100, 3), 50, np.uint8))

    submitted = []
    monkeypatch.setattr(window, "submit_frame", lambda bgr: submitted.append(bgr))
    try:
        window.active_source = "camera"
        window.camera_epoch = time.monotonic()
        window.video_frame(Frame(), source="file")
        assert not submitted
        window.video_frame(Frame(), source="camera")
        assert len(submitted) == 1
        window.camera_input.paused = True
        window.video_frame(Frame(), source="camera")
        assert len(submitted) == 1
    finally:
        window.close()


def test_selecting_output_part_keeps_whole_object_tracking_and_persists(application):
    window = MainWindow()
    try:
        image = np.zeros((180, 320, 3), np.uint8)
        image[40:120, 50:170] = np.random.default_rng(9).integers(20, 255, (80, 120, 3), np.uint8)
        subject = Rect(50 / 320, 40 / 180, 120 / 320, 80 / 180)
        window.latest_frame = image
        window.select_object(subject)
        spin(application, lambda: not window.busy and bool(window.input_view.boxes))
        original = window.video_worker.pipeline.object_tracker.box
        window.subject_region_combo.setCurrentIndex(1)
        window.motion_mode.setCurrentIndex(1)
        spin(application, lambda: not window.busy and bool(window.input_view.focus))
        assert window.video_worker.pipeline.object_tracker.box == original
        assert window.input_view.focus[0] == subject.region(Rect(0, 0, 0.5, 1))
        window.part_button.setChecked(True)
        window.select_input_region(subject.region(Rect(0.2, 0.1, 0.3, 0.6)))
        spin(application, lambda: not window.busy)
        assert window.settings.subject_region.x == pytest.approx(0.2)
        assert window.video_worker.pipeline.object_tracker.box == original
        profile = window.capture_profile()
        window.subject_region_combo.setCurrentIndex(0)
        window.apply_profile(profile)
        assert window.settings.subject_region.w == pytest.approx(0.3)
        assert window.settings.motion_mode == "responsive"
        assert not window.smoothing.isEnabled()
        assert window.scan_timer.interval() == 120
        with pytest.raises(ValueError, match="часть"):
            window.apply_profile(dict(profile, subject_region=[0.8, 0, 0.5, 1]))
        assert window.settings.subject_region.w == pytest.approx(0.3)
    finally:
        window.close()


def test_object_sample_plays_with_whole_target_and_partial_output(application):
    window = MainWindow()
    try:
        window.load_object_sample()
        spin(application, lambda: not window.output_view.image.isNull() and bool(window.input_view.focus), 20)
        spin(application, lambda: window.video_worker.pipeline.object_tracker.flow_updates >= 8, 20)
        window.pause_source()
        assert window.settings.subject_region == Rect(0, 0, 0.5, 1)
        assert window.settings.motion_mode == "responsive"
        assert window.input_view.focus[0].w == pytest.approx(window.input_view.boxes[0].w / 2)
        assert window.object_seed is not None
        spin(application, lambda: not window.busy)
        window.seek_to(0)
        spin(
            application, lambda: not window.seek_pending and not window.busy and bool(window.input_view.focus)
        )
        assert window.object_seed[1] == pytest.approx(0, abs=0.05)
    finally:
        window.close()


def test_auto_framing_finishes_on_paused_video_and_returns_to_part(application):
    window = MainWindow()
    try:
        window.load_object_sample()
        spin(application, lambda: bool(window.input_view.boxes) and not window.busy, 20)
        window.pause_source()
        spin(application, lambda: not window.busy)
        window.manual_frame(Rect(0.1, 0.1, 0.2, 0.4))
        window.auto_framing_enabled.setChecked(True)
        spin(application, lambda: not window.busy and window.last_framing.get("enabled"))
        assert not window.last_framing["whole"]
        original = window.video_worker.pipeline.object_tracker.box
        window.manual_frame(Rect(0.1, 0.1, 0.8, 0.8))
        spin(
            application,
            lambda: (
                not window.busy
                and window.last_framing.get("whole")
                and not window.last_framing.get("transitioning")
            ),
        )
        assert not window.input_view.focus
        assert window.video_worker.pipeline.object_tracker.box == original
        window.manual_frame(Rect(0.1, 0.1, 0.2, 0.4))
        spin(
            application,
            lambda: (
                not window.busy
                and not window.last_framing.get("whole")
                and not window.last_framing.get("transitioning")
            ),
        )
        assert window.input_view.focus[0].w == pytest.approx(original.w / 2)
        assert window.video_worker.pipeline.object_tracker.box == original
        window.remember_framing_threshold()
        assert window.frame_threshold.value() == pytest.approx(8)
    finally:
        window.close()


def test_adaptive_rule_profile_roundtrip_and_old_profiles_keep_fixed_mode(application):
    window = MainWindow()
    try:
        window.subject_region_combo.setCurrentIndex(1)
        window.auto_framing_enabled.setChecked(True)
        window.whole_when.setCurrentIndex(1)
        window.frame_threshold.setValue(35)
        window.add_scene_slot()
        window.frame_threshold.setValue(45)
        profile = window.capture_profile()
        window.apply_profile(profile)
        assert window.settings.framing.enabled
        assert window.settings.framing.whole_when == "smaller"
        assert window.settings.framing.threshold == pytest.approx(0.45)
        window.scene_combo.setCurrentIndex(0)
        assert window.settings.framing.threshold == pytest.approx(0.35)
        with pytest.raises(ValueError, match="автоматического"):
            window.apply_profile(dict(profile, framing={"threshold": float("nan")}))
        assert window.settings.framing.threshold == pytest.approx(0.35)
        old = dict(profile, version=5, slots=[], active_slot="")
        old.pop("framing")
        window.apply_profile(old)
        assert not window.settings.framing.enabled
    finally:
        window.close()


def test_old_file_sink_cannot_consume_selection_for_a_new_video(application, monkeypatch):
    window = MainWindow()
    try:
        previous = window.file_epoch
        window.load_object_sample()
        assert window.file_epoch != previous
        submitted = []
        monkeypatch.setattr(window, "video_frame", lambda *args, **kwargs: submitted.append(args))
        window.file_video_frame(object(), previous)
        assert not submitted
        assert window.pending_object_sample
        window.file_video_frame(object(), window.file_epoch)
        assert len(submitted) == 1
    finally:
        window.close()


def test_ambiguous_recovery_requires_explicit_candidate_click(application):
    from framepilot.tracking_validation import rigid_frame

    window = MainWindow()
    try:
        window.subject_mode.setCurrentIndex(window.subject_mode.findData("object"))
        image, first = rigid_frame(twin=True)
        second = Rect(first.x + 125 / 640, first.y, first.w, first.h)
        window.latest_frame = image
        window.latest_media_time = 1
        window.frame_ready(
            (
                window.session,
                image,
                None,
                [],
                1,
                "test fixture",
                None,
                {
                    "time": 1,
                    "tracking": {"state": "ambiguous", "candidates": 2, "basis": "test"},
                    "recovery": [first, second],
                },
            )
        )
        assert not window.recovery_button.isEnabled()
        assert "несколько похожих" in window.quality_label.text()
        window.confirm_recovery()
        assert window.object_seed is None
        window.select_face(second.center)
        spin(application, lambda: not window.busy)
        assert window.object_seed[0] == second
        assert window.input_view.boxes and iou(window.input_view.boxes[0], second) > 0.95
    finally:
        window.close()


def test_tracking_quality_is_in_status_and_cleared_for_new_source(application):
    window = MainWindow()
    try:
        evidence = {"state": "weak", "points": 9, "inlier_ratio": 0.7, "basis": "test"}
        window.update_tracking_quality(evidence)
        assert window.signal_report()["tracking_evidence"] == evidence
        assert "9" in window.quality_label.toolTip()
        assert "%" not in window.quality_label.text()
        window.arm_signal()
        assert window.last_tracking_quality == {}
        assert window.quality_label.isHidden()
    finally:
        window.close()
