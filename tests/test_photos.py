import os
import struct
import time
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import framepilot
from framepilot.app import MainWindow
from framepilot.geometry import Rect, iou
from framepilot.photos import read_photo
from framepilot.pipeline import Pipeline, Settings
from framepilot.recovery import read_recovery, write_recovery
from framepilot.widgets import to_array


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application):
    main = MainWindow()
    main.scan_timer.stop()
    try:
        yield main
    finally:
        main.close()
        application.processEvents()
    assert not main.photo_input.thread.isRunning()


def spin(app, predicate, timeout=10):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def photo(tmp_path):
    image = np.full((120, 200, 3), (25, 55, 210), np.uint8)
    image[:, 100:] = (30, 220, 35)
    path = tmp_path / "input.png"
    assert cv2.imwrite(str(path), image)
    return path


def test_jpeg_orientation_and_png_alpha_are_decoded(application, tmp_path):
    image = np.full((20, 40, 3), (30, 80, 160), np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    tiff = b"II" + struct.pack("<HIH", 42, 8, 1)
    tiff += struct.pack("<HHI", 0x112, 3, 1) + struct.pack("<H", 6) + b"\0\0" + b"\0" * 4
    exif = b"Exif\0\0" + tiff
    jpg = tmp_path / "rotated.jpg"
    jpg.write_bytes(
        encoded[:2].tobytes() + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + encoded[2:].tobytes()
    )
    assert read_photo(jpg).image.shape == (40, 20, 3)
    rgba = np.full((24, 32, 4), (20, 60, 200, 128), np.uint8)
    png = tmp_path / "alpha.png"
    cv2.imwrite(str(png), rgba)
    frame = read_photo(png)
    assert frame.original_size == (32, 24)
    np.testing.assert_allclose(frame.opacity, 128 / 255)
    np.testing.assert_array_equal(frame.image, rgba[:, :, :3])
    assert not frame.image.flags.writeable


def test_decode_limits_and_invalid_input(application, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        read_photo(bad)
    big = tmp_path / "large.png"
    cv2.imwrite(str(big), np.full((100, 5000, 3), 45, np.uint8))
    assert read_photo(big).image.shape[1] == 4096


def test_static_textureless_selection_and_opacity_do_not_require_motion_tracking():
    image = np.full((100, 100, 3), (20, 60, 200), np.uint8)
    opacity = np.full((100, 100), 0.5, np.float32)
    pipeline = Pipeline()
    settings = Settings(width=100, height=100, subject_mode="object", padding=0)
    output, box, _ = pipeline.process(
        image, 0, settings, Rect(0, 0, 1, 1), True, source_opacity=opacity, still=True
    )
    assert box == Rect(0, 0, 1, 1)
    np.testing.assert_allclose(output[50, 50], [10, 30, 100], atol=1)
    _, box, _ = pipeline.process(image, 0, settings, None, False, repaint=True, still=True)
    assert box == Rect(0, 0, 1, 1)


def test_photo_selection_frame_changes_output_and_save(application, window, photo, tmp_path):
    window.open_media(photo)
    spin(application, lambda: window.photo_ready and not window.busy)
    assert window.active_source == "photo"
    assert window.input_view.boxes == [Rect(0, 0, 1, 1)]
    window.select_object(Rect(0, 0, 0.5, 1))
    spin(application, lambda: not window.busy and window.input_view.boxes == [Rect(0, 0, 0.5, 1)])
    window.mode.setCurrentIndex(1)
    window.manual_frame(Rect(0.5, 0, 0.5, 1))
    spin(application, lambda: not window.busy and not window.render_pending)
    output = to_array(window.last_output)
    assert output[360, 960, 2] > 150 and output[360, 200].max() == 0
    for extension in ("png", "jpg"):
        destination = tmp_path / ("result." + extension)
        window.save_photo_image(destination)
        assert cv2.imread(str(destination)).shape == (720, 1280, 3)
    with pytest.raises(ValueError, match="перезаписывать"):
        window.save_photo_image(photo)


def test_static_photo_does_not_timeout_and_reaches_local_stream(application, window, photo):
    window.preferred_port = 0
    window.start_local_output()
    window.load_photo(photo)
    spin(application, lambda: window.photo_ready and not window.busy)
    window.signal_timeout.setValue(0.5)
    window.watch_signal()
    spin(application, lambda: window.publisher.hold_reason is None)
    until = time.monotonic() + 0.7
    spin(application, lambda: time.monotonic() >= until)
    window.watch_signal()
    assert window.publisher.hold_reason is None
    with urlopen(window.publisher.url.replace("/framepilot", "/stream.mjpg"), timeout=3) as response:
        assert response.readline().strip() == b"--frame"
        response.readline()
        length = int(response.readline().decode().split(":")[1])
        response.readline()
        image = cv2.imdecode(np.frombuffer(response.read(length), np.uint8), cv2.IMREAD_COLOR)
        assert image.max() > 150
    window.toggle_black_output()
    assert window.publisher.hold_reason == "emergency_black"
    window.toggle_black_output()
    spin(application, lambda: window.publisher.hold_reason is None)


def test_photo_face_detection_and_switching_sources(application, window, photo):
    portrait = Path(framepilot.__file__).parent / "assets/astronaut.png"
    window.load_photo(portrait)
    spin(application, lambda: window.photo_ready and not window.busy)
    window.photo_face()
    spin(
        application,
        lambda: not window.busy and bool(window.input_view.boxes) and window.input_view.boxes[0].area < 0.2,
    )
    assert window.input_view.boxes[0].area < 0.2
    window.load_photo(portrait)
    window.load_photo(photo)
    spin(application, lambda: window.photo_ready and not window.busy)
    assert window.photo_path == photo
    assert window.latest_frame.shape == (120, 200, 3)


def test_photo_scenario_and_recovery_keep_source_and_selection(application, window, photo, tmp_path):
    window.load_photo(photo)
    spin(application, lambda: window.photo_ready and not window.busy)
    subject = Rect(0.1, 0.2, 0.5, 0.5)
    window.select_object(subject)
    spin(
        application,
        lambda: (
            not window.busy
            and bool(window.input_view.boxes)
            and iou(window.input_view.boxes[0], subject) > 0.999
        ),
    )
    window.save_scenario("Photo selection")
    assert len(window.preferences.load()["scenarios"]) == 1
    snapshot = window.studio.snapshot()
    path = tmp_path / "recovery.json"
    write_recovery(path, snapshot)
    assert read_recovery(path)["source_kind"] == "photo"
    window.show_whole_photo()
    spin(application, lambda: not window.busy)
    window.apply_scenario()
    spin(
        application,
        lambda: (
            window.photo_ready
            and not window.busy
            and bool(window.input_view.boxes)
            and iou(window.input_view.boxes[0], subject) > 0.999
        ),
    )


def test_closed_local_output_stays_disabled_and_explicit_start_is_remembered(window):
    window.preferred_port = 0
    window.start_local_output()
    assert window.preferences.load()["local_output_enabled"] is True
    window.stop_local_output()
    assert window.preferences.load()["local_output_enabled"] is False


def test_photo_part_to_whole_animation_finishes_without_new_source_frames(application, window, photo):
    window.load_photo(photo)
    spin(application, lambda: window.photo_ready and not window.busy)
    window.subject_region_combo.setCurrentIndex(1)
    window.auto_framing_enabled.setChecked(True)
    window.frame_threshold.setValue(40)
    window.manual_frame(Rect(0.1, 0.1, 0.2, 0.3))
    spin(application, lambda: not window.busy and window.last_framing.get("whole") is False)
    selected = window.input_view.boxes[0]
    window.manual_frame(Rect(0.1, 0.1, 0.8, 0.8))
    spin(
        application,
        lambda: (
            not window.busy
            and window.last_framing.get("whole") is True
            and not window.last_framing.get("transitioning")
        ),
    )
    assert window.latest_media_time == 0
    assert window.input_view.boxes == [selected]
    assert window.last_framing["region"] == pytest.approx([0, 0, 1, 1])


def test_restarting_restores_only_enabled_local_output(application, tmp_path):
    first = MainWindow()
    first.preferred_port = 0
    first.start_local_output()
    url = first.publisher.url
    first.close()
    second = MainWindow()
    try:
        spin(application, lambda: second.publisher.url is not None)
        assert second.publisher.url == url
        assert not second.obs_connected and second.camera_input.camera is None
    finally:
        second.close()
        application.processEvents()
