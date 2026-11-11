import base64
import copy
import ctypes
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from framepilot.camera_identity import camera_label, is_obs_camera, obs_id
from framepilot.camera_name_dialog import CameraNameDialog
from framepilot.windows_camera_names import (
    CLASS_KEY,
    INSTANCE_KEY,
    OBS_GUID,
    CameraNameError,
    Registration,
    WindowsRegistry,
    change_names,
    load_original,
    registrations,
    remember_original,
    rename_targets,
    restore_targets,
    run_task,
    target_state,
    task_token,
)
from framepilot.windows_elevation import ShellExecuteInfo, helper_arguments


def original():
    return tuple(
        Registration(
            view, rf"C:\OBS\obs-virtualcam-module{view}.dll", "OBS Virtual Camera", "OBS Virtual Camera"
        )
        for view in (32, 64)
    )


class MemoryRegistry:
    def __init__(self, state=None, fail_on=()):
        self.state, self.calls, self.fail_on = state or original(), 0, set(fail_on)

    def snapshot(self):
        return self.state

    @contextmanager
    def updates(self, views):
        yield self

    def get(self, view, field):
        return getattr(next(r for r in self.state if r.view == view), field)

    def set(self, view, field, value):
        self.calls += 1
        if self.calls in self.fail_on:
            raise OSError("simulated registry write failure")
        self.state = tuple(replace(r, **{field: value}) if r.view == view else r for r in self.state)


@pytest.mark.parametrize("name", ["My Camera", "Камера рабочего стола", "Camera 🎬", "К" * 80])
def test_names_change_both_views_without_changing_module(name):
    registry = MemoryRegistry()
    change_names(registry, original(), rename_targets(original(), name))
    assert {r.friendly for r in registry.state} == {name}
    assert {r.class_name for r in registry.state} == {name}
    assert [r.module for r in registry.state] == [r.module for r in original()]


@pytest.mark.parametrize("name", ["", " ", " name", "name ", "a\nb", "a\0b", "a\tb", "К" * 81, "🎬" * 41])
def test_invalid_names_are_rejected(name):
    with pytest.raises(CameraNameError):
        rename_targets(original(), name)


def test_backup_keeps_first_names_after_multiple_renames_and_restore(tmp_path):
    path, registry = tmp_path / "original.json", MemoryRegistry()
    for name in ("First name", "Second name"):
        remember_original(path, registry.snapshot())
        change_names(registry, registry.snapshot(), rename_targets(registry.snapshot(), name))
    assert load_original(path) == original()
    change_names(registry, registry.snapshot(), restore_targets(registry.snapshot(), load_original(path)))
    assert registry.snapshot() == original()


def test_each_view_and_class_description_restores_its_actual_original_value(tmp_path):
    initial = (
        replace(original()[0], friendly="Original 32", class_name="Driver 32"),
        replace(original()[1], friendly="Original 64", class_name="Driver 64"),
    )
    registry, path = MemoryRegistry(initial), tmp_path / "original.json"
    remember_original(path, initial)
    change_names(registry, initial, rename_targets(initial, "Same new name"))
    change_names(registry, registry.snapshot(), restore_targets(registry.snapshot(), load_original(path)))
    assert registry.snapshot() == initial


def test_corrupt_backup_is_not_overwritten(tmp_path):
    path = tmp_path / "original.json"
    path.write_text("corrupted")
    with pytest.raises(CameraNameError, match="не будет перезаписана"):
        remember_original(path, original())
    assert path.read_text() == "corrupted"


@pytest.mark.parametrize("failed_write", [1, 2, 3, 4])
def test_failed_write_rolls_back_every_completed_value(failed_write):
    registry = MemoryRegistry(fail_on=[failed_write])
    with pytest.raises(CameraNameError) as error:
        change_names(registry, original(), rename_targets(original(), "New name"))
    assert error.value.code == 6 and registry.snapshot() == original()


def test_rollback_failure_is_reported():
    registry = MemoryRegistry(fail_on=[2, 3])
    with pytest.raises(CameraNameError) as error:
        change_names(registry, original(), rename_targets(original(), "New name"))
    assert error.value.code == 7 and registry.snapshot() != original()


def test_shared_registry_views_are_not_written_twice():
    class SharedRegistry(MemoryRegistry):
        def set(self, view, field, value):
            super().set(view, field, value)
            self.state = tuple(replace(r, **{field: value}) for r in self.state)

    registry = SharedRegistry()
    change_names(registry, original(), rename_targets(original(), "Shared name"))
    assert registry.calls == 2
    assert {r.friendly for r in registry.snapshot()} == {"Shared name"}


def test_rollback_preserves_another_process_edit():
    class ConcurrentRegistry(MemoryRegistry):
        def set(self, view, field, value):
            super().set(view, field, value)
            if self.calls == 1:
                self.state = tuple(
                    replace(r, class_name="External change") if r.view == 64 else r for r in self.state
                )

    registry = ConcurrentRegistry()
    with pytest.raises(CameraNameError) as error:
        change_names(registry, original(), rename_targets(original(), "New name"))
    assert error.value.code == 7
    assert registry.snapshot()[1].class_name == "External change"
    assert registry.snapshot()[0].friendly == original()[0].friendly


def test_changed_registration_blocks_all_writes():
    registry = MemoryRegistry((replace(original()[0], friendly="Changed by OBS"), original()[1]))
    with pytest.raises(CameraNameError) as error:
        change_names(registry, original(), rename_targets(original(), "New name"))
    assert error.value.code == 2 and registry.calls == 0


def test_removed_registration_is_not_recreated():
    registry = MemoryRegistry((original()[1],))
    change_names(registry, registry.snapshot(), rename_targets(registry.snapshot(), "Only x64"))
    assert len(registry.snapshot()) == 1 and registry.calls == 2
    assert set(restore_targets(registry.snapshot(), original())) == {"64"}


def test_payload_has_no_configurable_registry_paths_or_modules():
    token = task_token(original(), rename_targets(original(), 'User "name" & camera'))
    data = json.loads(base64.urlsafe_b64decode(token))
    assert registrations(data["expected"]) == original()
    assert target_state(original(), data["targets"])[0].friendly == 'User "name" & camera'
    for extra in ({"path": r"SOFTWARE\Other"}, {"InprocServer32": "other.dll"}):
        with pytest.raises(CameraNameError):
            task_token(original(), {"64": dict(friendly="x", class_name="x", **extra)})
    for changed in (
        dict(asdict(original()[0]), view=128),
        dict(asdict(original()[0]), module=r"C:\Other\driver.dll"),
    ):
        with pytest.raises(CameraNameError):
            registrations([changed])


def test_helper_decoder_applies_only_validated_name_payload(monkeypatch):
    import sys

    registry = MemoryRegistry()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("framepilot.windows_elevation.is_admin", lambda: True)
    monkeypatch.setattr("framepilot.windows_camera_names.WindowsRegistry", lambda: registry)
    assert run_task(task_token(original(), rename_targets(original(), "Helper name"))) == 0
    assert registry.snapshot()[1].friendly == "Helper name"
    before = registry.snapshot()
    assert run_task("invalid-base64!") == 4
    forged = base64.urlsafe_b64encode(
        json.dumps(
            {
                "version": 1,
                "expected": [asdict(r) for r in before],
                "targets": {"64": {"path": "OtherRegistryKey"}},
            }
        ).encode()
    ).decode()
    assert run_task(forged) == 4 and registry.snapshot() == before
    monkeypatch.setattr("framepilot.windows_elevation.is_admin", lambda: False)
    assert run_task(task_token(before, rename_targets(before, "No admin"))) == 5
    assert registry.snapshot() == before


class WinregMemory:
    HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER = "machine", "user"
    KEY_QUERY_VALUE, KEY_SET_VALUE = 1, 2
    KEY_WOW64_32KEY, KEY_WOW64_64KEY, REG_SZ = 0x200, 0x100, 1

    def __init__(self):
        self.data, self.writes, self.deny_view = {}, [], None
        for r in original():
            self.data["machine", CLASS_KEY + r"\InprocServer32", r.view] = {"": r.module}
            self.data["machine", CLASS_KEY, r.view] = {"": r.class_name}
            self.data["machine", INSTANCE_KEY, r.view] = {"FriendlyName": r.friendly}

    @contextmanager
    def OpenKey(self, root, path, zero, access):
        view = 64 if access & self.KEY_WOW64_64KEY else 32
        assert bool(access & self.KEY_WOW64_64KEY) != bool(access & self.KEY_WOW64_32KEY)
        if access & self.KEY_SET_VALUE and view == self.deny_view:
            raise PermissionError()
        key = root, path, view
        if key not in self.data:
            raise FileNotFoundError()
        yield key

    def QueryValueEx(self, key, name):
        return self.data[key][name], self.REG_SZ

    def SetValueEx(self, key, name, zero, kind, value):
        self.writes.append((key, name))
        self.data[key][name] = value


def test_registry_adapter_changes_only_existing_obs_name_values():
    module = WinregMemory()
    registry = WindowsRegistry(module)
    expected = registry.snapshot()
    change_names(registry, expected, rename_targets(expected, "Desk Camera"))
    assert len(module.writes) == 4
    assert {key[1] for key, _ in module.writes} == {CLASS_KEY, INSTANCE_KEY}
    assert {name for _, name in module.writes} == {"", "FriendlyName"}
    assert registry.snapshot()[1].friendly == "Desk Camera"


def test_permission_failure_does_not_leave_half_renamed_camera():
    module = WinregMemory()
    before = copy.deepcopy(module.data)
    module.deny_view = 64
    registry = WindowsRegistry(module)
    with pytest.raises(PermissionError):
        change_names(registry, registry.snapshot(), rename_targets(registry.snapshot(), "Desk Camera"))
    assert module.writes == [] and module.data == before


def test_per_user_override_is_reported_without_modification():
    module = WinregMemory()
    module.data["user", CLASS_KEY, 64] = {"": "User override"}
    with pytest.raises(CameraNameError, match="пользовательская регистрация"):
        WindowsRegistry(module).snapshot()
    assert module.writes == []


class Device:
    def __init__(self, name, identifier=b"physical-device"):
        self.name, self.identifier = name, identifier

    def description(self):
        return self.name

    def id(self):
        return self.identifier


def test_renamed_camera_is_identified_without_obs_in_name():
    guid = uuid.UUID(OBS_GUID)
    for raw in (str(guid).upper().encode(), str(guid).encode("utf-16-le"), guid.bytes_le):
        device = Device("Completely custom name", raw)
        assert obs_id(raw) and is_obs_camera(device)
        assert camera_label(device, original()) == "OBS Virtual Camera"
    aliases = (replace(original()[1], friendly="My Output"),)
    assert is_obs_camera(Device("My Output"), aliases)
    assert is_obs_camera(Device("OBS Virtual Camera"))
    assert not is_obs_camera(Device("Physical USB Camera"))


def test_win64_elevation_structure_and_arguments():
    assert ctypes.sizeof(ShellExecuteInfo) == 112
    assert helper_arguments("safe-token", frozen=True) == ["--camera-name-task", "safe-token"]
    assert helper_arguments("safe-token", frozen=False) == [
        "-B",
        "-m",
        "framepilot",
        "--camera-name-task",
        "safe-token",
    ]


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def spin(app, predicate):
    end = time.monotonic() + 3
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    assert predicate()


def test_ui_repeated_rename_restore_and_cancel(application, tmp_path, monkeypatch):
    monkeypatch.setattr("framepilot.camera_name_dialog.CameraInput.devices", lambda: [])
    registry, cancelled = MemoryRegistry(), [False]

    def runner(token, hwnd):
        if cancelled[0]:
            return 1223
        data = json.loads(base64.urlsafe_b64decode(token))
        change_names(registry, registrations(data["expected"]), data["targets"])
        return 0

    dialog = CameraNameDialog(tmp_path / "original.json", registry=registry, runner=runner)
    try:
        for name in ("Desk Camera", "Камера 2"):
            dialog.name.setText(name)
            dialog.apply_name()
            spin(application, lambda: not dialog.busy)
            assert registry.snapshot()[0].friendly == name
        dialog.restore_name()
        spin(application, lambda: not dialog.busy)
        assert registry.snapshot() == original()
        cancelled[0] = True
        dialog.name.setText("Cancelled name")
        dialog.apply_name()
        spin(application, lambda: not dialog.busy)
        assert registry.snapshot() == original() and "отменён" in dialog.status.text()
    finally:
        dialog.close()


def test_ui_duplicate_physical_camera_name_is_rejected(application, tmp_path, monkeypatch):
    monkeypatch.setattr("framepilot.camera_name_dialog.CameraInput.devices", lambda: [Device("USB Camera")])
    calls = []
    dialog = CameraNameDialog(
        tmp_path / "original.json", registry=MemoryRegistry(), runner=lambda *args: calls.append(args)
    )
    try:
        dialog.name.setText("USB Camera")
        dialog.apply_name()
        assert calls == [] and not dialog.busy and "другая камера" in dialog.status.text()
    finally:
        dialog.close()


def test_ui_reports_success_only_after_registry_readback(application, tmp_path, monkeypatch):
    monkeypatch.setattr("framepilot.camera_name_dialog.CameraInput.devices", lambda: [])
    registry = MemoryRegistry()
    dialog = CameraNameDialog(tmp_path / "original.json", registry=registry, runner=lambda *args: 0)
    try:
        dialog.name.setText("Not actually applied")
        dialog.apply_name()
        spin(application, lambda: not dialog.busy)
        assert registry.snapshot() == original()
        assert "не подтверждено" in dialog.status.text()
    finally:
        dialog.close()


def test_ui_cannot_close_while_elevated_job_is_active(application, tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr("framepilot.camera_name_dialog.CameraInput.devices", lambda: [])
    gate = threading.Event()

    def runner(*args):
        gate.wait(3)
        return 1223

    dialog = CameraNameDialog(tmp_path / "original.json", registry=MemoryRegistry(), runner=runner)
    dialog.show()
    try:
        dialog.name.setText("Pending name")
        dialog.apply_name()
        dialog.close()
        application.processEvents()
        assert dialog.isVisible() and dialog.busy
        gate.set()
        spin(application, lambda: not dialog.busy)
    finally:
        gate.set()
        spin(application, lambda: not dialog.busy)
        dialog.close()
