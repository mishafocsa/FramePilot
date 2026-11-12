"""Check a single published photo against real OBS Chromium, then restore its source."""

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

from framepilot import __version__
from framepilot.obs import SOURCE, OBSClient, websocket_config_path
from framepilot.output import LocalOutput


def check(report):
    settings = json.loads(websocket_config_path().read_text(encoding="utf-8"))
    client = OBSClient(settings.get("server_port", 4455), settings.get("server_password", ""))
    output = LocalOutput()
    original = None
    changed = False
    result = {
        "version": __version__,
        "passed": False,
        "scope": "Static MJPEG photos rendered by real OBS Browser Source",
    }
    try:
        client.connect()
        if client.request("GetStreamStatus").get("outputActive") or client.request("GetRecordStatus").get(
            "outputActive"
        ):
            raise RuntimeError("OBS is recording or streaming; photo check cancelled")
        client.status()  # Validate that the named source still belongs to FramePilot.
        if not client.request("GetSourceActive", {"sourceName": SOURCE}).get("videoShowing"):
            raise RuntimeError("FramePilot source is not visible; no scene was changed")
        original = client.request("GetInputSettings", {"inputName": SOURCE})["inputSettings"]
        url = output.start(0)
        first = np.full((360, 640, 3), (30, 70, 210), np.uint8)
        first[:, 320:] = (210, 90, 30)
        second = np.full_like(first, (45, 190, 60))
        second[:, 320:] = (130, 30, 180)
        output.publish(first)
        expected_first = cv2.imdecode(np.frombuffer(output.jpeg, np.uint8), cv2.IMREAD_COLOR)
        changed = True
        client.request(
            "SetInputSettings",
            {
                "inputName": SOURCE,
                "inputSettings": {"url": url, "width": 640, "height": 360, "shutdown": False},
                "overlay": True,
            },
        )
        client.request("PressInputPropertiesButton", {"inputName": SOURCE, "propertyName": "refreshnocache"})

        def observe(expected, timeout):
            deadline = time.monotonic() + timeout
            difference = float("inf")
            while time.monotonic() < deadline:
                data = client.request(
                    "GetSourceScreenshot",
                    {"sourceName": SOURCE, "imageFormat": "png", "imageWidth": 640, "imageHeight": 360},
                )["imageData"]
                image = cv2.imdecode(
                    np.frombuffer(base64.b64decode(data.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR
                )
                if image is not None and image.shape == expected.shape:
                    difference = float(np.abs(image.astype(np.int16) - expected.astype(np.int16)).mean())
                    if difference < 3:
                        return {"visible": True, "difference": round(difference, 3)}
                time.sleep(0.1)
            return {"visible": False, "difference": round(difference, 3) if np.isfinite(difference) else None}

        result["single_photo_after_browser_reload"] = observe(expected_first, 4)
        output.publish(second)
        expected_second = cv2.imdecode(np.frombuffer(output.jpeg, np.uint8), cv2.IMREAD_COLOR)
        result["changed_photo_without_playback"] = observe(expected_second, 3)
        client.request("PressInputPropertiesButton", {"inputName": SOURCE, "propertyName": "refreshnocache"})
        result["latest_photo_after_second_reload"] = observe(expected_second, 4)
        result["published_photos"] = output.frames
        result["passed"] = all(
            result[key]["visible"]
            for key in (
                "single_photo_after_browser_reload",
                "changed_photo_without_playback",
                "latest_photo_after_second_reload",
            )
        )
    finally:
        try:
            if changed and original is not None:
                client.request(
                    "SetInputSettings", {"inputName": SOURCE, "inputSettings": original, "overlay": False}
                )
                client.request(
                    "PressInputPropertiesButton", {"inputName": SOURCE, "propertyName": "refreshnocache"}
                )
                result["original_source_restored"] = True
        finally:
            client.close()
            output.stop()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    raise SystemExit(check(parser.parse_args().report))
