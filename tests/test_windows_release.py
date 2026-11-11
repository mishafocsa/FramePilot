import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def release():
    spec = importlib.util.spec_from_file_location(
        "windows_release", Path(__file__).parents[1] / "tools/windows_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executable(path, machine=0x8664):
    header = bytearray(128)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 0x3C, 64)
    header[64:68] = b"PE\0\0"
    struct.pack_into("<H", header, 68, machine)
    path.write_bytes(header)


def valid_report():
    return {
        "version": "0.15.2",
        "platform": "Windows",
        "passed": True,
        "checks": {
            "photo_face_and_save": True,
            "local_mjpeg_and_frame": True,
            "qt_video_playback": True,
            "face_backend": "YuNet · OpenCV DNN",
            "mp4_export_frames": 12,
        },
    }


@pytest.mark.parametrize(
    "field,value", [("platform", "Darwin"), ("version", "0.14.0"), ("passed", False), ("checks", None)]
)
def test_release_cannot_claim_success_from_wrong_or_failed_report(release, field, value):
    report = valid_report()
    report[field] = value
    with pytest.raises(ValueError):
        release.validate_windows_report(report, "0.15.2")


@pytest.mark.parametrize(
    "field,value",
    [
        ("photo_face_and_save", False),
        ("local_mjpeg_and_frame", False),
        ("qt_video_playback", False),
        ("face_backend", "Apple Vision"),
        ("mp4_export_frames", 0),
    ],
)
def test_incomplete_frozen_checks_block_the_release(release, field, value):
    report = valid_report()
    report["checks"][field] = value
    with pytest.raises(ValueError):
        release.validate_windows_report(report, "0.15.2")


def test_full_windows_report_is_accepted(release):
    release.validate_windows_report(valid_report(), "0.15.2")


def test_wrong_binary_does_not_replace_existing_archive(release, tmp_path):
    folder = tmp_path / "binary"
    folder.mkdir()
    (folder / "_internal").mkdir()
    (folder / "FramePilot.exe").write_bytes(b"a Mac executable cannot be shipped for Windows")
    target = tmp_path / "old.zip"
    target.write_bytes(b"previous working release")
    with pytest.raises(ValueError):
        release.package(folder, target, tmp_path, "0.15.2")
    assert target.read_bytes() == b"previous working release"


def test_32_bit_executable_is_not_shipped_as_x64(release, tmp_path):
    path = tmp_path / "example.exe"
    executable(path, 0x014C)
    with pytest.raises(ValueError, match="x64"):
        release.verify_exe(path)


def test_release_archive_contains_runtime_and_actual_checks(release, tmp_path):
    folder = tmp_path / "binary"
    (folder / "_internal").mkdir(parents=True)
    executable(folder / "FramePilot.exe")
    (folder / "_internal/runtime.txt").write_text("unit-test fixture")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "windows-check.json").write_text(json.dumps(valid_report()))
    target = release.package(folder, tmp_path / "unit-test.zip", reports, "0.15.2")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert "FramePilot/FramePilot.exe" in archive.namelist()
        assert "FramePilot/_internal/runtime.txt" in archive.namelist()
        assert json.loads(archive.read("FramePilot/QA/windows-check.json"))["passed"]
        assert b"Python is not required" in archive.read("FramePilot/START-HERE.txt")


def test_macos_cannot_be_reported_as_a_windows_build(release, monkeypatch):
    monkeypatch.setattr(release.sys, "platform", "darwin")
    monkeypatch.setattr(release, "run", lambda *a, **k: pytest.fail("A Mac must not run the Windows build"))
    assert release.main() == 2


def test_old_success_report_cannot_validate_a_failed_current_run(release, tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release.sys, "platform", "win32")
    monkeypatch.setattr(release.sysconfig, "get_platform", lambda: "win-amd64")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.15.2"\n')
    reports = tmp_path / "test-results/windows-release"
    reports.mkdir(parents=True)
    (reports / "windows-check.json").write_text(json.dumps(valid_report()))
    folder = tmp_path / "dist/FramePilot"
    (folder / "_internal").mkdir(parents=True)
    executable(folder / "FramePilot.exe")
    monkeypatch.setattr(release, "run", lambda *a, **k: None)
    assert release.main() == 1
    assert not (reports / "windows-check.json").exists()
    assert not list((tmp_path / "dist").glob("*.zip"))
