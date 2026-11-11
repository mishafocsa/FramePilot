import json

import pytest

from framepilot.obs import OBSActionError, OBSClient, authentication


class FakeOBS:
    def __init__(self, existing=False, conflict=False):
        self.queue = [{"op": 0, "d": {"authentication": {"salt": "salt", "challenge": "challenge"}}}]
        self.calls = []
        self.closed = False
        self.existing, self.conflict = existing, conflict

    def recv(self):
        return json.dumps(self.queue.pop(0))

    def send(self, text):
        packet = json.loads(text)
        if packet["op"] == 1:
            assert packet["d"]["authentication"] == authentication("test-password", "salt", "challenge")
            self.queue.append({"op": 2, "d": {"negotiatedRpcVersion": 1}})
            return
        data = packet["d"]
        self.calls.append((data["requestType"], data["requestData"]))
        responses = {
            "GetSceneList": {"scenes": [{"sceneName": "Existing user's scene"}]},
            "GetInputList": {"inputs": [{"inputName": "FramePilot Video"}] if self.existing else []},
            "GetInputSettings": {
                "inputKind": "browser_source",
                "inputSettings": {
                    "url": "https://unrelated.example/"
                    if self.conflict
                    else "http://127.0.0.1:18765/framepilot"
                },
            },
            "GetSceneItemList": {"sceneItems": [{"sourceName": "FramePilot Video", "sceneItemId": 42}]},
            "CreateInput": {"sceneItemId": 42},
            "CreateSceneItem": {"sceneItemId": 43},
            "GetVideoSettings": {"baseWidth": 1920, "baseHeight": 1080},
            "GetVirtualCamStatus": {"outputActive": False},
        }
        self.queue.append(
            {
                "op": 7,
                "d": {
                    "requestId": data["requestId"],
                    "requestStatus": {"result": True, "code": 100},
                    "responseData": responses.get(data["requestType"], {}),
                },
            }
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize("existing", [False, True])
def test_setup_uses_dedicated_scene_without_switching_live_output(existing):
    fake = FakeOBS(existing=existing)
    client = OBSClient(password="test-password", connector=lambda *a, **kw: fake)
    client.configure("http://127.0.0.1:18765/framepilot", 1280, 720)
    kinds = [kind for kind, _ in fake.calls]
    assert "CreateScene" in kinds
    assert ("SetInputSettings" if existing else "CreateInput") in kinds
    assert "SetCurrentProgramScene" not in kinds
    assert "StartVirtualCam" not in kinds
    assert "SetVideoSettings" not in kinds
    assert (
        "PressInputPropertiesButton",
        {"inputName": "FramePilot Video", "propertyName": "refreshnocache"},
    ) in fake.calls
    assert fake.closed


def test_explicit_activation_starts_camera_and_switches_scene():
    fake = FakeOBS()
    OBSClient(password="test-password", connector=lambda *a, **kw: fake).configure(
        "http://127.0.0.1:18765/framepilot", 1280, 720, activate=True
    )
    assert ("SetCurrentProgramScene", {"sceneName": "FramePilot"}) in fake.calls
    assert ("StartVirtualCam", {}) in fake.calls


def test_never_overwrites_unrelated_browser_source():
    fake = FakeOBS(existing=True, conflict=True)
    client = OBSClient(password="test-password", connector=lambda *a, **kw: fake)
    with pytest.raises(RuntimeError, match="другим содержимым"):
        client.configure("http://127.0.0.1:18765/framepilot", 1280, 720)
    assert not any(kind.startswith(("Create", "Set", "Start")) for kind, _ in fake.calls)
    assert fake.closed


def test_persistent_connection_syncs_dimensions_and_reports_real_camera_state():
    fake = FakeOBS()
    client = OBSClient(password="test-password", connector=lambda *a, **kw: fake)
    try:
        client.configure("http://127.0.0.1:18765/framepilot", 1280, 720, keep_open=True)
        assert not fake.closed
        assert client.password == ""
        fake.calls.clear()
        client.sync("http://127.0.0.1:18888/framepilot", 720, 1280, 24)
        updates = [data for kind, data in fake.calls if kind == "SetInputSettings"]
        assert updates[-1]["inputSettings"]["width"] == 720
        assert updates[-1]["inputSettings"]["height"] == 1280
        assert updates[-1]["inputSettings"]["fps"] == 24
        assert not any(
            kind in ("SetCurrentProgramScene", "SetVideoSettings", "StartVirtualCam", "SetSceneItemEnabled")
            for kind, _ in fake.calls
        )
        assert client.status() == {"connected": True, "virtual_camera": False, "source_showing": False}
    finally:
        client.close()


def test_sync_stops_when_source_is_replaced():
    fake = FakeOBS(existing=True, conflict=True)
    client = OBSClient(password="test-password", connector=lambda *a, **kw: fake)
    try:
        client.connect()
        with pytest.raises(RuntimeError, match="изменён"):
            client.sync("http://127.0.0.1:18765/framepilot", 1280, 720)
        assert not any(kind.startswith("Set") for kind, _ in fake.calls)
    finally:
        client.close()


@pytest.mark.parametrize("busy", ["GetStreamStatus", "GetRecordStatus"])
def test_busy_obs_cannot_be_activated_or_modified(busy):
    client = OBSClient()
    calls = []
    client.connect = lambda: None

    def request(kind, data=None):
        calls.append(kind)
        return {"outputActive": kind == busy}

    client.request = request
    with pytest.raises(OBSActionError, match="запись или трансляцию"):
        client.configure("http://127.0.0.1:18765/framepilot", 1280, 720, activate=True)
    assert not any(kind.startswith(("Create", "Set", "Start")) for kind in calls)


def test_diagnose_queries_actual_obs_parameters_without_exposing_scene_names():
    fake = FakeOBS()
    client = OBSClient(password="test-password", connector=lambda *a, **kw: fake)
    try:
        client.connect()
        result = client.diagnose()
        assert result["diagnostics"]["video"]["baseWidth"] == 1920
        assert result["diagnostics"]["source_url"] == "http://127.0.0.1:18765/framepilot"
        assert result["diagnostics"]["program_is_framepilot"] is False
        assert "Existing user's scene" not in json.dumps(result)
        assert "test-password" not in json.dumps(result)
    finally:
        client.close()
