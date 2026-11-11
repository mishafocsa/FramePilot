import pytest


@pytest.fixture(autouse=True)
def isolate_preferences(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEPILOT_CONFIG_DIR", str(tmp_path / "preferences"))
