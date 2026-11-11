"""Build on Windows x64; publish a ZIP only after tests and the frozen self-test pass."""

import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = (
    "test_geometry.py",
    "test_pipeline.py",
    "test_partial_target.py",
    "test_auto_framing.py",
    "test_probe.py",
    "test_output.py",
    "test_obs.py",
    "test_vision.py",
    "test_features.py",
    "test_windows_port.py",
    "test_windows_release.py",
    "test_camera_names.py",
    "test_audit_regressions.py",
    "test_reliability_0152.py",
    "test_connections.py",
    "test_signal_health.py",
    "test_photos.py",
)


def validate_windows_report(report, version):
    required = ("photo_face_and_save", "local_mjpeg_and_frame", "qt_video_playback")
    if not isinstance(report, dict) or not isinstance(report.get("checks"), dict):
        raise ValueError("Invalid Windows self-test report")
    checks = report.get("checks", {})
    if (
        report.get("version") != version
        or report.get("platform") != "Windows"
        or report.get("passed") is not True
        or not all(checks.get(name) is True for name in required)
        or not str(checks.get("face_backend", "")).startswith("YuNet")
        or checks.get("mp4_export_frames") != 12
    ):
        raise ValueError("Frozen Windows self-test did not confirm every required check")


def verify_exe(path):
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("Expected a Windows executable")
        stream.seek(0x3C)
        offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(offset)
        if stream.read(4) != b"PE\0\0" or stream.read(2) != b"\x64\x86":
            raise ValueError("Expected an x64 PE executable")


def run(command, log, env=None, timeout=900):
    print("Running:", subprocess.list2cmdline([str(c) for c in command]), flush=True)
    with Path(log).open("w", encoding="utf-8") as output:
        subprocess.run(
            command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=True
        )


def package(binary_directory, destination, reports, version):
    verify_exe(binary_directory / "FramePilot.exe")
    if not (binary_directory / "_internal").is_dir():
        raise ValueError("The bundled runtime directory is missing")
    # A failed build must never replace an existing release archive.
    descriptor, name = tempfile.mkstemp(prefix=".framepilot-windows-", suffix=".zip", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(binary_directory.rglob("*")):
                if path.is_file():
                    archive.write(path, Path("FramePilot") / path.relative_to(binary_directory))
            for path in sorted(reports.iterdir()):
                if path.is_file():
                    archive.write(path, Path("FramePilot/QA") / path.name)
            archive.writestr(
                "FramePilot/START-HERE.txt",
                f"FramePilot {version} for Windows x64\n\n"
                "Extract the entire FramePilot folder and launch FramePilot.exe.\n"
                "Keep _internal beside FramePilot.exe. Python is not required to run this build.\n"
                "OBS is required for OBS Virtual Camera output.\n\n"
                "Automated checks cover files, photos, YuNet, framing, local MJPEG and MP4.\n"
                "Physical cameras, window capture, OBS, UAC/registry changes and Android apps\n"
                "need separate interactive tests. See QA for the actual scope and results.\n",
            )
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("The release archive is damaged")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return destination


def main():
    if sys.platform != "win32" or sysconfig.get_platform() != "win-amd64":
        print(
            "Windows x64 and a Windows x64 Python interpreter are required; no EXE was built.",
            file=sys.stderr,
        )
        return 2
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    reports = ROOT / "test-results/windows-release"
    reports.mkdir(parents=True, exist_ok=True)
    source_env = dict(os.environ, FRAMEPILOT_VISION_BACKEND="yunet", QT_QPA_PLATFORM="offscreen")
    try:
        for name in (
            "dependencies.log",
            "tests.log",
            "tests.xml",
            "build.log",
            "self-test.log",
            "windows-check.json",
            "build-environment.json",
        ):
            (reports / name).unlink(missing_ok=True)
        run([sys.executable, "-m", "pip", "check"], reports / "dependencies.log")
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *(str(Path("tests") / name) for name in TEST_FILES),
                "--junitxml=" + str(reports / "tests.xml"),
            ],
            reports / "tests.log",
            source_env,
        )
        run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--workpath",
                ".build/windows",
                "FramePilot.spec",
            ],
            reports / "build.log",
        )
        binary_directory = ROOT / "dist/FramePilot"
        binary = binary_directory / "FramePilot.exe"
        verify_exe(binary)
        report_path = reports / "windows-check.json"
        report_path.unlink(missing_ok=True)
        frozen_env = dict(os.environ, FRAMEPILOT_VISION_BACKEND="yunet")
        frozen_env.pop("QT_QPA_PLATFORM", None)
        run(
            [str(binary), "--portable-self-test", "--report", str(report_path)],
            reports / "self-test.log",
            frozen_env,
            timeout=90,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_windows_report(report, version)
        (reports / "build-environment.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "scope": "Windows x64 build, selected cross-platform tests and frozen media/UI self-test; no hardware/OBS/UAC/Android validation",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        destination = ROOT / "dist" / f"FramePilot-{version}-Windows-x64.zip"
        package(binary_directory, destination, reports, version)
        print("Built and checked:", destination, flush=True)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, struct.error) as error:
        print("Windows release failed:", error, file=sys.stderr)
        print("No new release archive was published. Logs:", reports, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
