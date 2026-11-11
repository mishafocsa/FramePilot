import cv2
import numpy as np
import pytest

from framepilot.calibration import Calibration, received_preview
from framepilot.geometry import Rect
from framepilot.pipeline import Renderer, Settings
from framepilot.probe import ProbeHistory, VideoProbe


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_probe_reads_current_frame_crop_mirror_rotation_and_delay(rotation, mirror):
    probe = VideoProbe(1280, 720, token=81235)
    image, sequence = probe.frame(now=10)
    calibration = Calibration(Rect(0.03, 0.03, 0.94, 0.94), mirror, rotation=rotation)
    returned = received_preview(image, calibration)
    returned = cv2.resize(returned, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", returned, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    returned = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    observation = probe.observe(returned, observed_at=10.18, capture_started=10.16)
    assert observation.sequence == sequence
    assert observation.rotation == rotation
    assert observation.mirror == mirror
    assert observation.observed_delay_ms == pytest.approx(180)
    assert observation.capture_ms == pytest.approx(20)
    assert observation.crop.x == pytest.approx(0.03, abs=0.004)
    assert observation.crop.y == pytest.approx(0.03, abs=0.004)
    assert observation.crop.w == pytest.approx(0.94, abs=0.004)
    assert observation.crop.h == pytest.approx(0.94, abs=0.004)


def test_probe_requires_current_session_and_visible_camera_region():
    probe, other = VideoProbe(640, 360, token=42), VideoProbe(640, 360, token=73)
    image, _ = other.frame(now=10)
    with pytest.raises(ValueError, match="не найдена"):
        probe.observe(image, observed_at=10.1)
    image, _ = probe.frame(now=10)
    window = np.full((600, 1000, 3), 25, np.uint8)
    window[100:460, 150:790] = image
    result = probe.observe(window, Rect(0.15, 100 / 600, 0.64, 0.6), observed_at=10.1)
    assert result.sequence == 0
    assert result.crop.w == pytest.approx(1, abs=0.002)


def test_bad_counter_checksum_never_produces_a_latency_measurement():
    probe = VideoProbe(640, 360, token=42)
    image, _ = probe.frame(now=10)
    x, y, cell = probe.stamp_x, probe.stamp_y, probe.cell
    image[y + cell : y + cell * 2, x + cell : x + cell * 2] = (
        255 - image[y + cell : y + cell * 2, x + cell : x + cell * 2]
    )
    result = probe.observe(image, observed_at=10.1)
    assert result.sequence is None
    assert result.observed_delay_ms is None


def test_repeated_frames_do_not_inflate_delay_statistics_or_claim_dropped_frames():
    probe = VideoProbe(640, 360, token=42)
    image, _ = probe.frame(now=10)
    history = ProbeHistory()
    result = probe.observe(image, observed_at=10.1)
    assert history.add(result, now=10.1)
    assert not history.add(result, now=10.3)
    assert history.summary()["matched_unique_frames"] == 1
    assert history.summary()["repeated_observations"] == 1
    assert history.summary()["median_observed_delay_ms"] == pytest.approx(100)
    assert "dropped_frames" not in history.summary()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_video_compensation_keeps_original_object_orientation(rotation, mirror):
    image = np.zeros((120, 180, 3), np.uint8)
    image[30:90, 40:90] = (0, 0, 240)
    image[30:90, 90:140] = (240, 0, 0)
    image[30:45, 40:55] = (0, 240, 0)
    calibration = Calibration(Rect(0.1, 0.1, 0.8, 0.8), mirror, rotation=rotation)
    settings = Settings(width=400, height=300, padding=0, calibration=calibration)
    output = Renderer().render(image, Rect(40 / 180, 0.25, 100 / 180, 0.5), settings, 0)
    received = received_preview(output, calibration)
    left, right = received[:, : received.shape[1] // 2], received[:, received.shape[1] // 2 :]
    assert left[:, :, 2].mean() > left[:, :, 0].mean() + 60
    assert right[:, :, 0].mean() > right[:, :, 2].mean() + 60
    green = np.argwhere(received[:, :, 1] > 180)
    assert len(green)
    assert green[:, 0].mean() < received.shape[0] / 2
    assert green[:, 1].mean() < received.shape[1] / 2
