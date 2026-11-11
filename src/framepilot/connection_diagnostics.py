"""Explicit native-app test against an already enabled, local OBS WebSocket server."""

import json
import os
import platform
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QTimer

from . import __version__


def obs_self_test(app, report: Path):
    from .app import MainWindow
    from .connections import ConnectionWizard
    from .obs import OBSClient, websocket_config_path

    temporary = tempfile.TemporaryDirectory(prefix="framepilot-obs-check-")
    os.environ["FRAMEPILOT_CONFIG_DIR"] = temporary.name
    window = MainWindow()
    window.preferred_port = 0
    window.obs_timer.stop()
    window.show()
    wizard = ConnectionWizard(window)
    wizard.receiver.setCurrentIndex(wizard.receiver.findData("camera"))
    wizard.show()
    timer = QTimer(window)
    started = time.monotonic()
    state = {"phase": "connecting", "diagnosed": False, "last_devices": 0, "prepared": False}
    outcome = {
        "application": "FramePilot",
        "version": __version__,
        "machine": platform.machine(),
        "passed": False,
        "scope": "Local FramePilot output → real OBS Browser Source → OBS Virtual Camera → Qt receiver. Android is not tested.",
    }
    controller = None
    previous = None

    def finish(error=None):
        timer.stop()
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 2)
        outcome["connection"] = wizard.report()
        if error:
            outcome["error"] = error
        try:
            wizard.reject()
            if controller is not None and previous is not None and state["prepared"]:
                if previous.get("input_settings") is not None:
                    controller.request(
                        "SetInputSettings",
                        {
                            "inputName": "FramePilot Video",
                            "inputSettings": previous["input_settings"],
                            "overlay": False,
                        },
                    )
                    controller.request(
                        "PressInputPropertiesButton",
                        {
                            "inputName": "FramePilot Video",
                            "propertyName": "refreshnocache",
                        },
                    )
                if previous.get("item") is not None:
                    item = previous["item"]
                    controller.request(
                        "SetSceneItemTransform",
                        {
                            "sceneName": "FramePilot",
                            "sceneItemId": item["sceneItemId"],
                            "sceneItemTransform": item["sceneItemTransform"],
                        },
                    )
                    controller.request(
                        "SetSceneItemEnabled",
                        {
                            "sceneName": "FramePilot",
                            "sceneItemId": item["sceneItemId"],
                            "sceneItemEnabled": item["sceneItemEnabled"],
                        },
                    )
                if not previous["virtual_camera"]:
                    if controller.request("GetVirtualCamStatus").get("outputActive"):
                        controller.request("StopVirtualCam")
                controller.request("SetCurrentProgramScene", {"sceneName": previous["scene"]})
                if previous.get("input_settings") is None:
                    controller.request("RemoveInput", {"inputName": "FramePilot Video"})
                elif previous.get("item") is None and previous.get("scene_existed"):
                    items = controller.request("GetSceneItemList", {"sceneName": "FramePilot"})["sceneItems"]
                    for item in items:
                        if item["sourceName"] == "FramePilot Video":
                            controller.request(
                                "RemoveSceneItem",
                                {"sceneName": "FramePilot", "sceneItemId": item["sceneItemId"]},
                            )
                if not previous.get("scene_existed"):
                    controller.request("RemoveScene", {"sceneName": "FramePilot"})
                outcome["restored_obs_scene_and_camera"] = True
                outcome["restored_obs_source"] = True
        except Exception as error:
            outcome["restored_obs_scene_and_camera"] = False
            outcome["restoration_failure_type"] = type(error).__name__
            outcome["passed"] = False
        finally:
            if controller is not None:
                controller.close()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
            window.close()
            app.exit(0 if outcome["passed"] else 1)

    def poll():
        if time.monotonic() - started > 45:
            finish(
                "Timeout waiting for a current-session pattern from OBS Virtual Camera. Check camera permission and extension."
            )
            return
        if not window.obs_connected:
            if not window.obs_busy:
                finish("Local OBS connection or activation failed: " + window.obs_status.text())
            return
        if state["phase"] == "connecting":
            if window.obs_busy:
                return
            if not state["diagnosed"]:
                state["diagnosed"] = True
                wizard.diagnose_obs()
                return
            if wizard.obs_details is None:
                return
            if time.monotonic() - state["last_devices"] >= 1:
                state["last_devices"] = time.monotonic()
                wizard.refresh_cameras()
            if wizard.device_combo.currentData() is None:
                return
            wizard.start_probe()
            state["phase"] = "receiving"
            state["probe_started"] = time.monotonic()
        elif len(wizard.history.rows) >= 30 and time.monotonic() - state["probe_started"] >= 5:
            outcome["passed"] = wizard.history.stable_calibration() is not None
            finish(None if outcome["passed"] else "Unstable geometry from local virtual camera")

    try:
        config = websocket_config_path()
        settings = json.loads(config.read_text(encoding="utf-8"))
        if not settings.get("server_enabled"):
            raise ValueError("websocket_disabled")
        controller = OBSClient(settings.get("server_port", 4455), settings.get("server_password", ""))
        controller.connect()
        if controller.request("GetStreamStatus").get("outputActive") or controller.request(
            "GetRecordStatus"
        ).get("outputActive"):
            raise ValueError("obs_recording_or_streaming")
        previous = {
            "scene": controller.request("GetCurrentProgramScene")["currentProgramSceneName"],
            "virtual_camera": controller.request("GetVirtualCamStatus")["outputActive"],
        }
        inputs = controller.request("GetInputList")["inputs"]
        if any(item["inputName"] == "FramePilot Video" for item in inputs):
            previous["input_settings"] = controller.request(
                "GetInputSettings", {"inputName": "FramePilot Video"}
            )["inputSettings"]
        scenes = controller.request("GetSceneList")["scenes"]
        previous["scene_existed"] = any(scene["sceneName"] == "FramePilot" for scene in scenes)
        if previous["scene_existed"]:
            previous["item"] = next(
                (
                    item
                    for item in controller.request("GetSceneItemList", {"sceneName": "FramePilot"})[
                        "sceneItems"
                    ]
                    if item["sourceName"] == "FramePilot Video"
                ),
                None,
            )
        # Restore only after the entire original state is known and mutation can begin.
        state["prepared"] = True
        window.connect_obs(settings.get("server_port", 4455), settings.get("server_password", ""), True)
        settings.clear()
    except Exception as error:
        # Do not include raw transport/configuration exception text or credentials.
        outcome["startup_failure_type"] = type(error).__name__
        QTimer.singleShot(
            0, lambda: finish("Cannot prepare local OBS check; OBS must be idle with WebSocket enabled.")
        )
    else:
        timer.timeout.connect(poll)
        timer.start(100)
    result = app.exec()
    temporary.cleanup()
    return result
