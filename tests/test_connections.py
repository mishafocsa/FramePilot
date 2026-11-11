import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from framepilot.app import MainWindow
from framepilot.connections import ConnectionWizard, safe_service_url
from framepilot.geometry import Rect
from framepilot.telemetry import FrameTelemetry
from framepilot.widgets import to_array


@pytest.fixture(scope="module")
def application():
    yield QApplication.instance() or QApplication([])


def spin(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_wizard_internal_roundtrip_never_claims_obs_or_android_and_restores_video(application):
    main = MainWindow()
    wizard = ConnectionWizard(main)
    try:
        main.preferred_port = 0
        original = np.full((720, 1280, 3), (10, 30, 70), np.uint8)
        main.show_processed(original)
        main.demo.timer.stop()
        main.demo.animate = False
        wizard.show()
        wizard.start_probe()
        assert main.connection_probe is not None
        spin(application, lambda: len(wizard.history.rows) >= 3)
        assert wizard.history.stable_calibration() is not None
        assert not wizard.apply_calibration_button.isEnabled()  # Window chrome is not camera crop.
        data = wizard.report()
        assert data["return_scope"] == "framepilot_internal_demo"
        assert data["stages"]["returned_pattern"] == "observed_current_session"
        assert data["stages"]["obs_api"] == data["stages"]["android_camera"] == "not_verified"
        assert data["virtual_camera_callback_fps"] is None
        wizard.set_visible_roi(main.demo.slot()[0])
        spin(application, lambda: len(wizard.history.rows) >= 3)
        assert wizard.apply_calibration_button.isEnabled()
        wizard.stop_probe()
        assert main.connection_probe is None and main.scan_timer.isActive()
        np.testing.assert_array_equal(to_array(main.last_output), original)
    finally:
        wizard.reject()
        main.close()
        application.processEvents()
    assert not wizard.thread.isRunning()


def test_roi_change_discards_old_measurements_and_profile_contains_no_secrets(application):
    main = MainWindow()
    wizard = ConnectionWizard(main)
    try:
        main.preferred_port = 0
        main.show_processed(np.zeros((720, 1280, 3), np.uint8))
        wizard.start_probe()
        wizard.capture_timer.stop()
        image, _ = wizard.probe.frame()
        observation = wizard.probe.observe(image)
        previous = wizard.generation
        wizard.set_visible_roi(Rect(0.1, 0.1, 0.8, 0.8))
        wizard.observed(previous, observation, image)
        assert not wizard.history.geometry
        assert not wizard.apply_calibration_button.isEnabled()
        wizard.obs_password.setText("not-in-any-report")
        wizard.service_url.setText("https://example.test/emulator?token=private-token#secret-fragment")
        wizard.destination.setCurrentIndex(wizard.destination.findData("online"))
        wizard.save_connection()
        saved = main.preferences.load()["connection_profile"]
        assert saved["service"] == "https://example.test/emulator"
        combined = json.dumps([saved, wizard.report()])
        assert all(
            secret not in combined for secret in ("private-token", "secret-fragment", "not-in-any-report")
        )
        main.padding.setValue(0.7)
        wizard.restore_video_profile()
        assert main.padding.value() == pytest.approx(saved["profile"]["padding"])
        assert wizard.report()["stages"]["android_camera"] == "not_verified"
    finally:
        wizard.reject()
        main.close()
        application.processEvents()


def test_service_url_rejects_credentials_and_local_protocols():
    for value in ("file:///etc/passwd", "javascript:alert(1)", "https://user:password@example.test"):
        with pytest.raises(ValueError):
            safe_service_url(value)


def test_telemetry_separates_repaints_busy_skips_and_recent_source_fps():
    telemetry = FrameTelemetry()
    telemetry.received, telemetry.accepted, telemetry.busy_skips = 3, 2, 1
    telemetry.complete(2, True, now=10)
    telemetry.complete(4, False, now=10.04)
    telemetry.complete(1, False, now=10.05, repaint=True)
    stats = telemetry.summary(now=10.1)
    assert stats["recent_processing_fps"] == 25
    assert stats["completed_frames"] == 2 and stats["unconfirmed_target_frames"] == 1
    assert stats["repaint_jobs"] == 1 and stats["skipped_while_busy"] == 1
    assert telemetry.summary(now=20)["recent_processing_fps"] is None


def test_rotation_profile_validation_is_atomic_and_older_profiles_default_zero(application):
    main = MainWindow()
    try:
        profile = main.capture_profile()
        profile["calibration"]["rotation"] = 270
        main.apply_profile(profile)
        assert main.settings.calibration.rotation == 270
        invalid = main.capture_profile()
        invalid["calibration"]["rotation"] = 45
        with pytest.raises(ValueError, match="поворот"):
            main.apply_profile(invalid)
        assert main.settings.calibration.rotation == 270
        profile["calibration"].pop("rotation")
        profile["version"] = 6
        main.apply_profile(profile)
        assert main.settings.calibration.rotation == 0
    finally:
        main.close()


def test_closing_main_window_stops_wizard_worker(application):
    main = MainWindow()
    wizard = ConnectionWizard(main)
    main.close()
    application.processEvents()
    assert not wizard.thread.isRunning()
    assert main.connection_wizard is None
