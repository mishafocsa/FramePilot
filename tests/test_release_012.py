import os
import subprocess
import threading
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from framepilot.analysis import AnalysisJob, analyze_clip
from framepilot.app import MainWindow
from framepilot.appearance import install_appearance
from framepilot.automation import Automation, Cue, Sample, source_identity
from framepilot.calibration import Calibration
from framepilot.exporter import ExportJob, export_video
from framepilot.geometry import Rect
from framepilot.masks import FrameMask
from framepilot.media_check import probe_video
from framepilot.object_geometry import QuadSelection, quad_bounds, region_quad
from framepilot.pipeline import Pipeline, Renderer, Settings
from framepilot.recovery import read_recovery, write_recovery
from framepilot.tracking_validation import card_picture, rigid_frame


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    install_appearance(app)
    return app


@pytest.fixture
def window(application):
    w = MainWindow()
    w.show()
    application.processEvents()
    try:
        yield w
    finally:
        w.close()
        application.processEvents()
    assert not w.studio.media.thread.isRunning()
    assert not w.studio.recovery.thread.isRunning()


def spin(app, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def rotating_video(tmp_path):
    path = tmp_path / "rotating.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    assert writer.isOpened()
    for i in range(45):
        writer.write(rigid_frame(i, angle=i)[0])
    writer.release()
    return path


@pytest.mark.parametrize("angle", [0, 30, 60, 90, 120])
def test_surface_part_follows_rotation_without_exposing_other_half(angle):
    image = np.zeros((360, 640, 3), np.uint8)
    matrix = cv2.getRotationMatrix2D((60, 45), -angle, 1)
    matrix[:, 2] += [260, 135]
    card = card_picture()
    image = cv2.warpAffine(card, matrix, (640, 360))
    quad = cv2.transform(np.float32([[[0, 0], [120, 0], [120, 90], [0, 90]]]), matrix)[0] / [640, 360]
    renderer = Renderer()
    settings = Settings(
        width=320,
        height=240,
        subject_mode="object",
        surface_lock=True,
        subject_region=Rect(0, 0, 0.5, 1),
        padding=0,
    )
    output = renderer.render(image, quad_bounds(quad), settings, 0, snap=True, object_quad=quad)
    red = (output[:, :, 2] > 180) & (output[:, :, 1] < 110)
    green = (output[:, :, 1] > 205) & (output[:, :, 2] < 120) & (output[:, :, 0] < 100)
    assert red.sum() > 200
    assert green.sum() < 10
    np.testing.assert_allclose(
        renderer.last_surface_quad, region_quad(quad, settings.subject_region), atol=1e-6
    )
    assert renderer.last_report["surface_locked"]


def test_actual_tracker_surface_and_cached_analysis_survive_rotation(rotating_video):
    pipeline = Pipeline()
    settings = Settings(
        width=320,
        height=240,
        subject_mode="object",
        surface_lock=True,
        subject_region=Rect(0, 0, 0.5, 1),
        padding=0,
    )
    for i in range(45):
        image, box = rigid_frame(i, angle=i)
        output, subject, _ = pipeline.process(image, i / 30, settings, box if i == 0 else None, i == 0)
        assert subject is not None
        assert ((output[:, :, 1] > 205) & (output[:, :, 2] < 120) & (output[:, :, 0] < 100)).sum() < 15
    _, tracks = analyze_clip(
        AnalysisJob(rotating_video, settings, rigid_frame()[1], 0, 30), threading.Event()
    )
    assert all(s.quad is not None for s in tracks[0].samples)
    automation = Automation(tracks=tracks)
    assert Automation.from_record(automation.record()) == automation
    image, _ = rigid_frame(30, angle=30)
    cached = Pipeline()
    cached.process(image, 1, replace(settings, automation=automation), None, True)
    assert cached.renderer.last_report["surface_locked"]
    assert cached.renderer.last_surface_quad is not None


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_oriented_selection_and_cue_preserve_part_under_receiver_calibration(rotation, mirror):
    transform = cv2.getRotationMatrix2D((60, 45), -90, 1)
    transform[:, 2] += [260, 135]
    image = cv2.warpAffine(card_picture(), transform, (640, 360))
    quad = cv2.transform(np.float32([[[0, 0], [120, 0], [120, 90], [0, 90]]]), transform)[0] / [640, 360]
    points = tuple(tuple(float(v) for v in point) for point in quad)
    cue = Cue(1, "single", quad_bounds(quad), Rect(0, 0, 0.5, 1), 0, 1, points)
    automation = Automation.from_record(Automation((cue,)).record())
    settings = Settings(
        width=320,
        height=240,
        subject_mode="object",
        surface_lock=True,
        calibration=Calibration(rotation=rotation, mirror=mirror),
        automation=automation,
        padding=0,
    )
    for selection in (None, QuadSelection(points)):
        pipeline = Pipeline()
        output, subject, _ = pipeline.process(image, 1, settings, selection, True)
        assert subject is not None
        assert ((output[:, :, 2] > 180) & (output[:, :, 1] < 110)).sum() > 200
        assert ((output[:, :, 1] > 205) & (output[:, :, 2] < 120) & (output[:, :, 0] < 100)).sum() < 15


def test_old_analysis_without_geometry_never_substitutes_axis_aligned_part():
    from framepilot.automation import Trajectory

    image, box = rigid_frame()
    old = Sample.from_record([[box.x, box.y, box.w, box.h], "stable", None])
    automation = Automation(tracks=(Trajectory("single", "object", 0, 30, (old,)),))
    settings = Settings(
        width=320, height=240, subject_mode="object", surface_lock=True, automation=automation, loss="black"
    )
    pipeline = Pipeline()
    result, subject, _ = pipeline.process(image, 0, settings, None, True)
    assert subject is not None
    assert pipeline.renderer.last_report["state"] == "surface_unavailable"
    assert not result.any()


def test_composition_reports_clipping_and_limited_resolution(tmp_path):
    image = np.full((100, 100, 3), 200, np.uint8)
    renderer = Renderer()
    settings = Settings(width=200, height=100, padding=0, fill=True)
    renderer.render(image, Rect(0, 0, 1, 1), settings, 0, snap=True)
    assert renderer.last_report["visible_fraction"] == pytest.approx(0.5, abs=0.02)
    assert renderer.last_report["warnings"]
    renderer = Renderer()
    renderer.render(
        image, Rect(0, 0, 1, 1), replace(settings, width=400, height=400, fill=False), 0, snap=True
    )
    assert renderer.last_report["visible_fraction"] > 0.98
    assert renderer.last_report["upscale"] == 4
    assert len(renderer.last_report["warnings"]) == 1
    mask = np.zeros((100, 100, 4), np.uint8)
    mask[:, :50, 3] = 255
    path = tmp_path / "half.png"
    cv2.imwrite(str(path), mask)
    renderer = Renderer()
    renderer.render(
        image,
        Rect(0, 0, 1, 1),
        Settings(width=100, height=100, padding=0, frame_mask=FrameMask("png", image=str(path))),
        0,
        snap=True,
    )
    assert renderer.last_report["visible_fraction"] == pytest.approx(0.5, abs=0.02)


def test_batch_exports_independent_snapshots_and_rejects_changed_source(
    application, window, rotating_video, tmp_path
):
    track = rigid_frame()[1]
    batch = window.studio.batch
    settings = Settings(width=160, height=120, subject_mode="object", surface_lock=True)
    first = ExportJob(rotating_video, tmp_path / "one.mp4", settings, 15, audio=False, selection=track)
    second = replace(
        first, destination=tmp_path / "two.mp4", settings=replace(settings, frame_mask=FrameMask("circle"))
    )
    batch.add(first)
    batch.add(second)
    with pytest.raises(ValueError, match="такой файл"):
        batch.add(second)
    window.padding.setValue(0.9)
    batch.start()
    spin(application, lambda: not batch.running and not window.export_busy)
    assert [e.state for e in batch.entries] == ["done", "done"]
    assert first.destination.exists() and second.destination.exists()
    batch.add(replace(first, destination=tmp_path / "changed.mp4"))
    with rotating_video.open("ab") as stream:
        stream.write(b"changed")
    batch.start()
    spin(application, lambda: not batch.running and not window.export_busy)
    assert batch.entries[-1].state == "error"
    assert not (tmp_path / "changed.mp4").exists()


def test_recovery_restores_file_recipe_and_selection_paused(application, window, rotating_video):
    w = window
    w.load_video(rotating_video)
    spin(application, lambda: w.latest_frame is not None and not w.busy)
    w.pause_source()
    w.seek_to(0)
    spin(application, lambda: not w.seek_pending and not w.busy)
    w.select_object(rigid_frame()[1])
    spin(application, lambda: not w.busy and bool(w.input_view.boxes))
    w.studio.surface_lock.setChecked(True)
    w.subject_region_combo.setCurrentIndex(1)
    spin(application, lambda: not w.busy)
    w.player.play()
    spin(application, lambda: w.latest_media_time > 0.4)
    w.pause_source()
    spin(application, lambda: not w.busy)
    snapshot = w.studio.snapshot()
    assert snapshot is not None
    w.studio.recovery.save(snapshot)
    spin(application, lambda: not w.studio.recovery.busy)
    data = read_recovery(w.studio.recovery.path)
    assert data["profile"]["surface_lock"]
    draft = {
        **snapshot,
        "automation": replace(
            snapshot["automation"], override_time=w.latest_media_time, override_slot="single"
        ),
    }
    write_recovery(w.studio.recovery.path, draft)
    assert read_recovery(w.studio.recovery.path)["automation"].override_time == w.latest_media_time
    other = MainWindow()
    try:
        other.show()
        spin(application, lambda: other.studio.recovery_data is not None)
        assert other.camera_input.camera is None and not other.studio.screen.running
        other.studio.restore_recovery()
        spin(
            application,
            lambda: (
                other.latest_frame is not None and not other.busy and not other.studio.pause_after_recovery
            ),
        )
        assert other.video_path == rotating_video
        assert other.player.playbackState() == other.player.PlaybackState.PausedState
        assert other.settings.subject_region == Rect(0, 0, 0.5, 1)
        assert other.settings.surface_lock
        assert other.object_seed is not None
        assert isinstance(other.object_seed[0], QuadSelection)
        restored = other.video_worker.pipeline.object_tracker.quad
        expected = snapshot["selections"]["single"]["value"]
        np.testing.assert_allclose(restored, expected, atol=0.002)
    finally:
        other.close()
        application.processEvents()
    before = w.studio.recovery.path.read_bytes()
    bad = {**snapshot, "expected_identity": {}}
    with pytest.raises(ValueError, match="изменился"):
        write_recovery(w.studio.recovery.path, bad)
    assert w.studio.recovery.path.read_bytes() == before


@pytest.mark.parametrize(
    "extension,codec,extra",
    [
        ("mp4", "libx264", []),
        ("mov", "libx265", ["-x265-params", "pools=1:frame-threads=1:log-level=error"]),
        ("mov", "prores_ks", ["-profile:v", "2"]),
        ("mkv", "mpeg4", []),
        ("avi", "mjpeg", []),
        ("webm", "libvpx-vp9", ["-deadline", "realtime"]),
        ("mp4", "libaom-av1", ["-cpu-used", "8", "-crf", "40"]),
    ],
)
def test_common_container_codec_pairs_decode_in_preview_and_export(
    application, window, tmp_path, extension, codec, extra
):
    path = tmp_path / (codec + "." + extension)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=15",
            "-t",
            "0.4",
            "-c:v",
            codec,
            "-threads",
            "1",
            *extra,
            str(path),
        ],
        check=True,
        timeout=20,
        capture_output=True,
    )
    info = probe_video(path)
    assert info["identity"] == source_identity(path)
    assert info["size"] == [160, 96]
    window.load_video(path)
    spin(application, lambda: window.latest_frame is not None)
    window.pause_source()
    assert window.studio.media.preview
    assert window.player.error() == window.player.Error.NoError
    destination = tmp_path / "exported.mp4"
    count = export_video(
        ExportJob(path, destination, Settings(width=160, height=96, loss="original"), 15, audio=False),
        threading.Event(),
    )
    assert count == 6


def test_invalid_file_has_explicit_decode_error(tmp_path):
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="не смог прочитать"):
        probe_video(path)


def test_loaded_quality_report_remains_usable_in_compact_window(application, window):
    image, subject = rigid_frame()
    window.latest_frame = image
    window.select_object(subject)
    spin(application, lambda: not window.busy and bool(window.input_view.boxes))
    window.studio.surface_lock.setChecked(True)
    window.subject_region_combo.setCurrentIndex(1)
    spin(application, lambda: not window.busy)
    window.editor.navigate(1)
    window.resize(1120, 740)
    window.editor.refresh_status()
    application.processEvents()
    assert window.width() == 1120
    assert window.last_composition["warnings"]
    assert window.editor.deck.cards["output"].detail.toolTip()
    for card in window.editor.deck.cards.values():
        assert card.x() >= 0 and card.x() + card.width() <= window.editor.deck.width()
