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
