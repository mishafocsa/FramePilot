import subprocess
import threading
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

import framepilot
from framepilot.exporter import ExportCancelled, ExportJob, export_video
from framepilot.framing import FramingRule
from framepilot.geometry import Rect
from framepilot.pipeline import SceneSlot, Settings


@pytest.fixture
def source_video(tmp_path):
    source = tmp_path / "portrait-with-audio.mp4"
    portrait = Path(framepilot.__file__).parent / "assets/astronaut.png"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(portrait),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "0.8",
            "-r",
            "15",
            "-vf",
            "scale=256:256",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    return source


def test_actual_framed_h264_mp4_preserves_audio_and_duration(source_video, tmp_path):
    destination = tmp_path / "framed.mp4"
    settings = Settings(width=320, height=240)
    job = ExportJob(source_video, destination, settings, fps=15, start=0.2, audio=True)
    progress = []
    count = export_video(job, threading.Event(), progress.append)
    assert 8 <= count <= 10
    assert progress[-1] == 100
    frames = imageio_ffmpeg.read_frames(str(destination), pix_fmt="bgr24")
    metadata = next(frames)
    first = next(frames)
    frames.close()
    assert metadata["size"] == (320, 240)
    assert "h264" in metadata["codec"]
    assert metadata["duration"] == pytest.approx(0.6, abs=0.12)
    assert np.frombuffer(first, np.uint8).mean() > 10
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(destination), "-f", "null", "-"],
        capture_output=True,
        timeout=20,
    )
    assert probe.returncode == 0
    assert b"Audio: aac" in probe.stderr


def test_cancel_mid_export_keeps_existing_destination(source_video, tmp_path):
    destination = tmp_path / "existing.mp4"
    destination.write_bytes(b"existing-file-must-survive")
    cancel = threading.Event()
    job = ExportJob(source_video, destination, Settings(width=160, height=120), fps=15, audio=False)
    with pytest.raises(ExportCancelled):
        export_video(job, cancel, lambda progress: cancel.set())
    assert destination.read_bytes() == b"existing-file-must-survive"
    assert list(tmp_path.glob(".framepilot-*")) == []


def test_native_person_cutout_and_two_slot_composition_export(source_video, tmp_path):
    destination = tmp_path / "scene.mp4"
    slots = (
        SceneSlot("face", Rect(0, 0, 0.45, 1)),
        SceneSlot("upper", Rect(0.55, 0, 0.45, 1), subject_mode="upper"),
    )
    settings = Settings(
        width=400,
        height=240,
        slots=slots,
        active_slot="face",
        remove_background=True,
        background_color="#00ff00",
    )
    count = export_video(
        ExportJob(source_video, destination, settings, fps=15, audio=True), threading.Event()
    )
    assert 11 <= count <= 13
    video = imageio_ffmpeg.read_frames(str(destination), pix_fmt="bgr24")
    metadata = next(video)
    first = np.frombuffer(next(video), np.uint8).reshape(240, 400, 3)
    video.close()
    assert "h264" in metadata["codec"]
    np.testing.assert_allclose(first[100, 200], (0, 255, 0), atol=8)
    assert np.std(first[:, :160]) > 30 and np.std(first[:, 240:]) > 30


@pytest.mark.parametrize("auto_whole", [False, True])
def test_mp4_partial_and_automatic_whole_views(tmp_path, auto_whole):
    import cv2

    source, destination = tmp_path / "two-halves.avi", tmp_path / "half.mp4"
    image = np.zeros((180, 320, 3), np.uint8)
    image[50:130, 80:140] = (0, 0, 240)
    image[50:130, 140:200] = (240, 0, 0)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 180))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(image)
    writer.release()
    settings = Settings(
        width=320,
        height=240,
        subject_mode="object",
        padding=0,
        subject_region=Rect(0, 0, 0.5, 1),
        motion_mode="responsive",
        fill=True,
        framing=FramingRule(enabled=auto_whole, threshold=0.1, transition=0),
    )
    count = export_video(
        ExportJob(
            source,
            destination,
            settings,
            fps=15,
            audio=False,
            selection=Rect(0.25, 50 / 180, 120 / 320, 80 / 180),
        ),
        threading.Event(),
    )
    assert count >= 5
    reader = imageio_ffmpeg.read_frames(str(destination), pix_fmt="bgr24")
    next(reader)
    first = np.frombuffer(next(reader), np.uint8).reshape(240, 320, 3)
    reader.close()
    assert first[:, :, 2].max() > 200
    if auto_whole:
        assert first[:, :, 0].max() > 200
    else:
        assert first[120, :, 0].max() < 25


def test_cannot_overwrite_source(source_video):
    with pytest.raises(ValueError, match="исходное"):
        export_video(ExportJob(source_video, source_video, Settings()), threading.Event())


def test_silent_input_and_manual_object_export(source_video, tmp_path):
    from framepilot.geometry import Rect

    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-c:v",
            "copy",
            "-an",
            str(silent),
        ],
        check=True,
        timeout=20,
    )
    destination = tmp_path / "object.mp4"
    count = export_video(
        ExportJob(
            silent,
            destination,
            Settings(width=160, height=120, subject_mode="object"),
            fps=15,
            audio=True,
            selection=Rect(0.3, 0.1, 0.3, 0.4),
        ),
        threading.Event(),
    )
    assert 11 <= count <= 13
    video = imageio_ffmpeg.read_frames(str(destination), pix_fmt="bgr24")
    metadata = next(video)
    assert metadata["size"] == (160, 120)
    assert np.frombuffer(next(video), np.uint8).mean() > 10
    video.close()
