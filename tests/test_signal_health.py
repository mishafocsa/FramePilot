import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication

from framepilot.app import MainWindow
from framepilot.geometry import Rect
from framepilot.output import LocalOutput
from framepilot.signal_health import SignalMonitor, SignalPolicy
from framepilot.widgets import to_array


def decoded(output):
    return cv2.imdecode(np.frombuffer(output.jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_static_scene_with_advancing_timestamps_is_live_and_pause_has_no_timeout():
    monitor = SignalMonitor()
    monitor.arm(now=0)
    assert monitor.poll(True, 2, now=0)["state"] == "waiting"
    for index in range(1, 100):
        stamp = index / 30
        monitor.received(stamp, now=stamp)
        assert monitor.complete(stamp, stamp, 2, now=stamp + 0.02)
        assert monitor.poll(True, 2, now=stamp + 0.03)["state"] == "live"
    assert monitor.poll(False, 2, now=100)["state"] == "paused"
    assert monitor.poll(True, 2, now=110)["state"] == "live"  # Resume gets a fresh grace period.
    assert monitor.poll(True, 2, now=113)["state"] == "input_stalled"


def test_repeated_source_timestamp_or_repaint_cannot_mask_a_stall():
    monitor = SignalMonitor()
    monitor.arm(now=0)
    monitor.poll(True, 2, now=0)
    monitor.received(1, now=0)
    monitor.complete(1, 0, 2, now=0.01)
    monitor.received(1, now=3)
    assert not monitor.complete(1, 3, 2, now=3)
    assert not monitor.complete(2, 3, 2, now=3, repaint=True)
    assert monitor.poll(True, 2, now=3)["state"] == "input_stalled"
    monitor.received(2, now=3.1)
    assert not monitor.complete(2, 0, 2, now=3.2)  # Delayed old job cannot reopen the output.
    assert monitor.complete(2, 3.1, 2, now=3.2)
    assert monitor.poll(True, 2, now=3.2)["state"] == "live"


def test_input_frames_and_processing_have_independent_timeout_and_calibration_is_suspended():
    monitor = SignalMonitor()
    monitor.arm(now=0)
    monitor.poll(True, 2, now=0)
    monitor.received(1, now=0)
    monitor.complete(1, 0, 2, now=0.1)
    monitor.received(100, now=3)
    assert monitor.poll(True, 2, now=3)["state"] == "processing_stalled"
    assert monitor.poll(False, 2, suspended=True, now=100)["state"] == "calibrating"
    monitor.received(101, now=100.1)
    monitor.complete(101, 100.1, 2, now=100.2)
    assert monitor.poll(True, 2, now=100.3)["state"] == "live"


@pytest.mark.parametrize(
    "value",
    [None, [], {"timeout": True}, {"timeout": float("nan")}, {"timeout": 99}, {"fallback": "original"}],
)
def test_invalid_stored_signal_settings_fall_back_to_defaults(value):
    assert SignalPolicy.from_record(value) == SignalPolicy()


def test_output_gate_rejects_expired_jobs_and_jobs_encoding_across_hold_and_release(monkeypatch):
    output = LocalOutput()
    output.enabled.set()
    old = np.full((80, 120, 3), 240, np.uint8)
    black = np.zeros_like(old)
    output.publish(old, deadline=time.monotonic() - 1)
    assert output.jpeg is None
    started, finish = threading.Event(), threading.Event()
    encode = cv2.imencode

    def delayed(extension, image, *args):
        if image is old:
            started.set()
            assert finish.wait(5)
        return encode(extension, image, *args)

    monkeypatch.setattr(cv2, "imencode", delayed)
    thread = threading.Thread(target=lambda: output.publish(old))
    thread.start()
    try:
        assert started.wait(5)
        output.hold(black, "emergency_black")
        output.reset(5, clear=True)
        assert not decoded(output).any()
        output.release()
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not decoded(output).any(), "An encoding started before the hold must not overwrite its frame"
    output.publish(old, 5)
    assert decoded(output).mean() > 230


@pytest.fixture(scope="module")
def application():
    yield QApplication.instance() or QApplication([])


def result(main, image, stamp, repaint=False):
    return (
        main.session,
        image,
        Rect(0.2, 0.2, 0.3, 0.4),
        [],
        1,
        "test fixture",
        None,
        {"time": stamp, "submitted_at": time.monotonic(), "repaint": repaint},
    )


def test_black_button_survives_new_frames_source_reset_and_server_restart(application):
    main = MainWindow()
    try:
        main.preferred_port = 0
        main.show_processed(np.full((720, 1280, 3), (0, 180, 0), np.uint8))
        main.start_local_output()
        main.toggle_black_output()
        assert main.publisher.hold_reason == "emergency_black"
        main.invalidate_frame(clear=True)
        main.frame_ready(result(main, np.full((720, 1280, 3), (0, 0, 200), np.uint8), 1))
        assert not to_array(main.last_output).any()
        assert not decoded(main.publisher).any()
        main.stop_local_output()
        main.start_local_output()
        assert not decoded(main.publisher).any()
        main.toggle_black_output()
        assert main.publisher.hold_reason is None
        assert to_array(main.last_output)[100, 100, 2] == 200
        assert decoded(main.publisher)[100, 100, 2] > 190
    finally:
        main.close()


@pytest.mark.parametrize("fallback", ["black", "freeze"])
def test_stall_fallback_is_fixed_until_a_fresh_result_and_preferences_persist(
    application, monkeypatch, fallback
):
    main = MainWindow()
    try:
        main.preferred_port = 0
        original = np.full((720, 1280, 3), 70, np.uint8)
        main.show_processed(original)
        main.signal_fallback.setCurrentIndex(main.signal_fallback.findData(fallback))
        main.start_local_output()
        monkeypatch.setattr(main.player, "playbackState", lambda: QMediaPlayer.PlaybackState.PlayingState)
        main.signal_monitor.arm()
        main.watch_signal()
        now = time.monotonic()
        main.signal_monitor.received(1, now=now - 4)
        main.signal_monitor.complete(1, now - 4, 2, now=now - 3.9)
        main.watch_signal()
        assert main.signal_state["state"] == "input_stalled"
        expected = 0 if fallback == "black" else 70
        assert to_array(main.last_output).mean() == expected
        main.show_processed(np.full_like(original, 140))
        assert to_array(main.last_output).mean() == expected
        main.resolution.setCurrentIndex(2)
        assert main.last_output.width() == 720 and main.last_output.height() == 1280
        assert to_array(main.last_output).mean() == expected
        main.signal_monitor.received(2)
        main.frame_ready(result(main, np.full((1280, 720, 3), 160, np.uint8), 2))
        assert main.publisher.hold_reason is None
        assert decoded(main.publisher).mean() == pytest.approx(160, abs=1)
        main.save_preferences()
        assert main.preferences.load()["signal_protection"]["fallback"] == fallback
    finally:
        main.close()


def test_source_change_waits_for_real_frame_and_blocks_old_session_and_repaint(application):
    main = MainWindow()
    try:
        image = np.full((720, 1280, 3), 180, np.uint8)
        main.show_processed(image)
        late = result(main, image, 1)
        main.grid.setChecked(True)
        main.arm_signal()
        assert not main.grid.isChecked()
        main.invalidate_frame(clear=True)
        main.frame_ready(late)
        assert main.publisher.hold_reason == "waiting"
        main.frame_ready(result(main, image, 2, repaint=True))
        assert main.publisher.hold_reason == "waiting"
        assert not to_array(main.last_output).any()
        main.signal_monitor.received(3)
        main.frame_ready(result(main, image, 3))
        assert main.publisher.hold_reason is None
        assert to_array(main.last_output).mean() == 180
    finally:
        main.close()
