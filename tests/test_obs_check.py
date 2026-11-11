import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import framepilot.obs


@pytest.mark.parametrize("startup_fails", [False, True])
def test_external_check_restores_exact_obs_configuration_even_after_failure(
    tmp_path, monkeypatch, startup_fails
):
    spec = importlib.util.spec_from_file_location(
        "check_obs", Path(__file__).parents[1] / "tools/check_obs.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = tmp_path / "Library/Application Support/obs-studio/plugin_config/obs-websocket/config.json"
    config.parent.mkdir(parents=True)
    original = b'{"server_enabled": false, "server_port": 4455, "auth_required": true, "server_password": "existing-secret"}'
    config.write_bytes(original)
    monkeypatch.setattr(helper, "running_obs", lambda: [])
    monkeypatch.setattr(helper.plistlib, "loads", lambda _: {"CFBundleExecutable": "OBS"})
    read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: b"plist" if path.name == "Info.plist" else read_bytes(path)
    )
    clock = iter(range(1000))
    monkeypatch.setattr(helper.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(helper.time, "sleep", lambda _: None)

    class Process:
        pid = 99999999

        def __init__(self):
            self.waits = 0
            self.terminations = 0

        def poll(self):
            return None if self.waits < 2 else 0

        def terminate(self):
            self.terminations += 1

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("OBS", timeout)
            return 0

    process = Process()
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *args, **kwargs: process)

    def check_app(*args, **kwargs):
        active = json.loads(config.read_text())
        assert active["auth_required"] and active["server_enabled"]
        assert active["server_password"] == "existing-secret"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", check_app)

    class Client:
        def __init__(self, *args):
            pass

        def connect(self):
            if startup_fails:
                raise RuntimeError("not ready")

        def request(self, kind, data=None):
            return {"outputActive": False, "currentProgramSceneName": "Original"}

        def close(self):
            pass

    monkeypatch.setattr(framepilot.obs, "OBSClient", Client)
    if startup_fails:
        with pytest.raises(RuntimeError, match="did not become available"):
            helper.run(tmp_path / "report.json")
    else:
        assert helper.run(tmp_path / "report.json") == 0
    assert config.read_bytes() == original
    assert process.waits == 2 and process.terminations == 2
