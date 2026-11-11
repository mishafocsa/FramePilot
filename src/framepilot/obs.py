"""OBS WebSocket v5 integration. Only changes a dedicated FramePilot scene/source."""

import base64
import hashlib
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

import websocket

SCENE = "FramePilot"
SOURCE = "FramePilot Video"


def websocket_config_path(platform=None, home=None, environ=None):
    platform = sys.platform if platform is None else platform
    home = Path.home() if home is None else Path(home)
    environ = os.environ if environ is None else environ
    relative = Path("obs-studio/plugin_config/obs-websocket/config.json")
    if platform == "win32":
        return Path(environ.get("APPDATA") or home / "AppData/Roaming") / relative
    if platform == "darwin":
        return home / "Library/Application Support" / relative
    return Path(environ.get("XDG_CONFIG_HOME") or home / ".config") / relative


class OBSActionError(RuntimeError):
    """A safe, actionable message without credentials or transport exception text."""


def authentication(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


class OBSClient:
    def __init__(
        self, port: int = 4455, password: str = "", cancel: threading.Event | None = None, connector=None
    ):
        self.port, self.password = port, password
        self.cancel = cancel or threading.Event()
        self.connector = connector or websocket.create_connection
        self.socket = None

    def connect(self):
        self.socket = self.connector(f"ws://127.0.0.1:{self.port}", timeout=3, suppress_origin=True)
        hello = json.loads(self.socket.recv())
        if hello.get("op") != 0:
            raise RuntimeError("Сервер не поддерживает OBS WebSocket v5")
        data = {"rpcVersion": 1, "eventSubscriptions": 0}
        auth = hello.get("d", {}).get("authentication")
        if auth:
            data["authentication"] = authentication(self.password, auth["salt"], auth["challenge"])
        self.socket.send(json.dumps({"op": 1, "d": data}))
        identified = json.loads(self.socket.recv())
        if identified.get("op") != 2:
            raise RuntimeError("OBS не принял подключение. Проверьте пароль WebSocket.")
        self.password = ""

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def request(self, kind: str, data: dict | None = None) -> dict:
        if self.cancel.is_set():
            raise RuntimeError("Подключение OBS отменено")
        identifier = str(uuid.uuid4())
        self.socket.send(
            json.dumps(
                {"op": 6, "d": {"requestType": kind, "requestId": identifier, "requestData": data or {}}}
            )
        )
        while not self.cancel.is_set():
            response = json.loads(self.socket.recv())
            payload = response.get("d", {})
            if response.get("op") == 7 and payload.get("requestId") == identifier:
                status = payload.get("requestStatus", {})
                if not status.get("result"):
                    raise RuntimeError(f"OBS {kind}: {status.get('comment') or status.get('code')}")
                return payload.get("responseData", {})
        raise RuntimeError("Подключение OBS отменено")

    def configure(
        self,
        url: str,
        width: int,
        height: int,
        activate: bool = False,
        keep_open: bool = False,
        fps: int = 30,
    ) -> str:
        successful = False
        try:
            self.connect()
            if activate and (
                self.request("GetStreamStatus").get("outputActive")
                or self.request("GetRecordStatus").get("outputActive")
            ):
                raise OBSActionError("OBS ведёт запись или трансляцию. Сцена и камера не переключены.")
            scenes = self.request("GetSceneList")["scenes"]
            inputs = self.request("GetInputList")["inputs"]
            existing = next((item for item in inputs if item["inputName"] == SOURCE), None)
            if existing:
                settings = self.request("GetInputSettings", {"inputName": SOURCE})
                previous = urlparse(settings.get("inputSettings", {}).get("url", ""))
                if (
                    settings.get("inputKind") != "browser_source"
                    or previous.hostname != "127.0.0.1"
                    or previous.path != "/framepilot"
                ):
                    raise RuntimeError(
                        f"Источник «{SOURCE}» уже используется другим содержимым. Переименуйте его в OBS."
                    )
            if not any(item["sceneName"] == SCENE for item in scenes):
                self.request("CreateScene", {"sceneName": SCENE})
            browser = {
                "url": url,
                "width": width,
                "height": height,
                "fps": fps,
                "fps_custom": True,
                "is_local_file": False,
                "shutdown": False,
            }
            if existing:
                self.request(
                    "SetInputSettings", {"inputName": SOURCE, "inputSettings": browser, "overlay": True}
                )
                items = self.request("GetSceneItemList", {"sceneName": SCENE})["sceneItems"]
                item = next((item for item in items if item["sourceName"] == SOURCE), None)
                if item is None:
                    item_id = self.request("CreateSceneItem", {"sceneName": SCENE, "sourceName": SOURCE})[
                        "sceneItemId"
                    ]
                else:
                    item_id = item["sceneItemId"]
            else:
                item_id = self.request(
                    "CreateInput",
                    {
                        "sceneName": SCENE,
                        "inputName": SOURCE,
                        "inputKind": "browser_source",
                        "inputSettings": browser,
                        "sceneItemEnabled": True,
                    },
                )["sceneItemId"]
            video = self.request("GetVideoSettings")
            self.request(
                "SetSceneItemTransform",
                {
                    "sceneName": SCENE,
                    "sceneItemId": item_id,
                    "sceneItemTransform": {
                        "positionX": video["baseWidth"] / 2,
                        "positionY": video["baseHeight"] / 2,
                        "alignment": 0,
                        "boundsType": "OBS_BOUNDS_SCALE_INNER",
                        "boundsAlignment": 0,
                        "boundsWidth": video["baseWidth"],
                        "boundsHeight": video["baseHeight"],
                    },
                },
            )
            self.request(
                "SetSceneItemEnabled", {"sceneName": SCENE, "sceneItemId": item_id, "sceneItemEnabled": True}
            )
            # An unchanged URL can still contain Chromium's previous connection-error page.
            self.request(
                "PressInputPropertiesButton", {"inputName": SOURCE, "propertyName": "refreshnocache"}
            )
            if activate:
                self.request("SetCurrentProgramScene", {"sceneName": SCENE})
                if not self.request("GetVirtualCamStatus")["outputActive"]:
                    self.request("StartVirtualCam")
            successful = True
            return "Сцена FramePilot готова" + (
                " · виртуальная камера включена" if activate else " · выберите её в OBS"
            )
        finally:
            if not keep_open or not successful:
                self.close()

    def sync(self, url: str, width: int, height: int, fps: int = 30):
        """Update our source only. Do not alter the live scene or reactivate a hidden item."""
        source = self.request("GetInputSettings", {"inputName": SOURCE})
        address = urlparse(source.get("inputSettings", {}).get("url", ""))
        if (
            source.get("inputKind") != "browser_source"
            or address.hostname != "127.0.0.1"
            or address.path != "/framepilot"
        ):
            raise RuntimeError("Источник FramePilot изменён в OBS; автоматическая синхронизация остановлена")
        self.request(
            "SetInputSettings",
            {
                "inputName": SOURCE,
                "inputSettings": {
                    "url": url,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "fps_custom": True,
                },
                "overlay": True,
            },
        )
        items = self.request("GetSceneItemList", {"sceneName": SCENE})["sceneItems"]
        item = next((item for item in items if item["sourceName"] == SOURCE), None)
        if item is None:
            raise RuntimeError("Источник FramePilot удалён из сцены")
        video = self.request("GetVideoSettings")
        self.request(
            "SetSceneItemTransform",
            {
                "sceneName": SCENE,
                "sceneItemId": item["sceneItemId"],
                "sceneItemTransform": {
                    "positionX": video["baseWidth"] / 2,
                    "positionY": video["baseHeight"] / 2,
                    "alignment": 0,
                    "boundsType": "OBS_BOUNDS_SCALE_INNER",
                    "boundsAlignment": 0,
                    "boundsWidth": video["baseWidth"],
                    "boundsHeight": video["baseHeight"],
                },
            },
        )

    def status(self):
        source = self.request("GetInputSettings", {"inputName": SOURCE})
        address = urlparse(source.get("inputSettings", {}).get("url", ""))
        if (
            source.get("inputKind") != "browser_source"
            or address.hostname != "127.0.0.1"
            or address.path != "/framepilot"
        ):
            raise RuntimeError("Источник FramePilot удалён или заменён")
        return {
            "connected": True,
            "virtual_camera": bool(self.request("GetVirtualCamStatus")["outputActive"]),
            "source_showing": bool(
                self.request("GetSourceActive", {"sourceName": SOURCE}).get("videoShowing")
            ),
        }

    def diagnose(self):
        state = self.status()
        source = self.request("GetInputSettings", {"inputName": SOURCE}).get("inputSettings", {})
        video = self.request("GetVideoSettings")
        state["diagnostics"] = {
            "obs_version": self.request("GetVersion").get("obsVersion"),
            "source_url": source.get("url"),
            "source_width": source.get("width"),
            "source_height": source.get("height"),
            "source_fps": source.get("fps"),
            "video": {
                key: video.get(key)
                for key in (
                    "baseWidth",
                    "baseHeight",
                    "outputWidth",
                    "outputHeight",
                    "fpsNumerator",
                    "fpsDenominator",
                )
            },
            "program_is_framepilot": self.request("GetCurrentProgramScene").get("currentProgramSceneName")
            == SCENE,
            "stream_active": bool(self.request("GetStreamStatus").get("outputActive")),
            "record_active": bool(self.request("GetRecordStatus").get("outputActive")),
        }
        return state
