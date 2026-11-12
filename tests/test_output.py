import json
import socket
import urllib.error
import urllib.request
from urllib.parse import urljoin

import cv2
import numpy as np
import pytest

from framepilot import __version__
from framepilot.output import LocalOutput


def test_local_output_streams_encoded_frame_and_stops():
    output = LocalOutput()
    try:
        url = output.start(0)
        with urllib.request.urlopen(url, timeout=3) as response:
            assert response.status == 200
            assert b"/stream.mjpg" in response.read()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "missing", timeout=3)
        image = np.full((120, 160, 3), (10, 30, 200), np.uint8)
        output.reset(3)
        output.publish(image, 3)
        sequence = output.sequence
        output.publish(np.zeros_like(image), 2)
        assert output.sequence == sequence, "Old sessions must never reach OBS"
        with urllib.request.urlopen(urljoin(url, "/stream.mjpg"), timeout=3) as response:
            assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
            assert response.readline().strip() == b"--frame"
            assert response.readline().strip() == b"Content-Type: image/jpeg"
            length = int(response.readline().split(b":")[1])
            assert response.readline().strip() == b""
            data = response.read(length)
            decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == image.shape
            np.testing.assert_allclose(decoded[50, 50], image[50, 50], atol=4)
    finally:
        output.stop()
    assert output.server is None
    assert output.thread is None


def test_output_reuses_active_address_and_can_restart():
    output = LocalOutput()
    try:
        first = output.start(0)
        assert output.start(0) == first
        output.stop()
        assert output.start(0).startswith("http://127.0.0.1:")
    finally:
        output.stop()


def test_single_photo_is_fully_delimited_before_a_second_photo_is_published():
    output = LocalOutput()
    colors = ((10, 30, 200), (30, 190, 50))
    try:
        url = output.start(0)
        output.publish(np.full((60, 80, 3), colors[0], np.uint8))
        with urllib.request.urlopen(urljoin(url, "/stream.mjpg"), timeout=1) as response:
            assert response.readline().strip() == b"--frame"
            for index, color in enumerate(colors):
                assert response.readline().strip() == b"Content-Type: image/jpeg"
                length = int(response.readline().split(b":")[1])
                assert response.readline().strip() == b""
                image = cv2.imdecode(np.frombuffer(response.read(length), np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_allclose(image[30, 40], color, atol=4)
                assert response.readline() == b"\r\n"
                # A browser can finish this JPEG even when the next photo never arrives.
                assert response.readline().strip() == b"--frame"
                if index == 0:
                    output.publish(np.full((60, 80, 3), colors[1], np.uint8))
        assert output.frames == 2
    finally:
        output.stop()


def test_static_photo_keeps_streaming_without_inflating_processed_frame_count():
    output = LocalOutput()
    try:
        url = output.start(0)
        output.publish(np.full((60, 80, 3), (25, 160, 75), np.uint8))
        sequence = output.sequence
        original = output.jpeg
        with urllib.request.urlopen(urljoin(url, "/stream.mjpg"), timeout=2) as response:
            assert response.readline().strip() == b"--frame"
            for _ in range(3):
                assert response.readline().strip() == b"Content-Type: image/jpeg"
                length = int(response.readline().split(b":")[1])
                assert response.readline().strip() == b""
                assert response.read(length) == original
                assert response.readline() == b"\r\n"
                assert response.readline().strip() == b"--frame"
        assert output.frames == 1
        assert output.sequence == sequence
    finally:
        output.stop()


def test_busy_port_falls_back_to_owned_server_and_reports_identity():
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    port = occupied.getsockname()[1]
    output = LocalOutput()
    try:
        url = output.start(port)
        assert output.server.server_port != port
        with urllib.request.urlopen(urljoin(url, "/api/status"), timeout=3) as response:
            status = json.load(response)
        assert status["application"] == "FramePilot"
        assert status["version"] == __version__
        output.update_health({"state": "paused", "emergency_black": False})
        with urllib.request.urlopen(urljoin(url, "/api/status"), timeout=3) as response:
            status = json.load(response)
        assert status["signal"]["state"] == "paused"
        assert status["hold_reason"] is None
        assert output.url == url
    finally:
        output.stop()
        occupied.close()
