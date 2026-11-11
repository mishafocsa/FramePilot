"""Rename only the standard OBS DirectShow registration. Never installs/replaces a driver."""

import base64
import json
import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PureWindowsPath

OBS_GUID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
VIDEO_CATEGORY = "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
CLASS_KEY = rf"SOFTWARE\Classes\CLSID\{OBS_GUID}"
INSTANCE_KEY = rf"SOFTWARE\Classes\CLSID\{VIDEO_CATEGORY}\Instance\{OBS_GUID}"
FIELDS = {"friendly": (INSTANCE_KEY, "FriendlyName"), "class_name": (CLASS_KEY, "")}
MODULES = {"obs-virtualcam-module32.dll", "obs-virtualcam-module64.dll", "obs-virtualcam-module.dll"}
MAX_NAME = 80
MESSAGES = {
    0: "Имя камеры изменено. Перезапустите принимающие приложения; иногда нужен перезапуск FramePilot.",
    2: "Регистрация камеры изменилась. Нажмите «Обновить» и повторите действие.",
    3: "Стандартная OBS Virtual Camera не найдена или её регистрация отличается от поддерживаемой.",
    4: "Неверное имя или параметры переименования.",
    5: "Нет доступа к регистрации камеры. Windows должна разрешить изменение с правами администратора.",
    6: "Переименование не завершено; сделанные изменения отменены.",
    7: "Не удалось полностью отменить изменения. Проверьте текущие имена и используйте сохранённое исходное имя.",
    1223: "Запрос прав администратора отменён. Переименование не выполнялось.",
}


class CameraNameError(RuntimeError):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or MESSAGES.get(code, "Windows не завершила изменение имени камеры."))


def valid_name(value, limit=MAX_NAME):
    if not isinstance(value, str) or not value or value != value.strip() or not value.isprintable():
        raise CameraNameError(4, "Введите непустое имя без переводов строк и пробелов по краям.")
    if len(value.encode("utf-16-le")) // 2 > limit:
        raise CameraNameError(4, f"Имя должно содержать не более {limit} символов UTF-16.")
    return value


@dataclass(frozen=True)
class Registration:
    view: int
    module: str
    friendly: str
    class_name: str


def registrations(records):
    if not isinstance(records, list) or not 1 <= len(records) <= 2:
        raise CameraNameError(4)
    result = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"view", "module", "friendly", "class_name"}:
            raise CameraNameError(4)
        if type(record["view"]) is not int or record["view"] not in (32, 64):
            raise CameraNameError(4)
        module = record["module"]
        if not isinstance(module, str) or len(module) > 4096 or not module.isprintable():
            raise CameraNameError(4)
        if PureWindowsPath(module).name.lower() not in MODULES:
            raise CameraNameError(3, "Регистрация OBS указывает на неизвестный модуль камеры.")
        result.append(
            Registration(
                record["view"],
                module,
                valid_name(record["friendly"], 512),
                valid_name(record["class_name"], 512),
            )
        )
    if len({item.view for item in result}) != len(result):
        raise CameraNameError(4)
    return tuple(sorted(result, key=lambda item: item.view))


class WindowsRegistry:
    def __init__(self, module=None):
        if module is None:
            if sys.platform != "win32":
                raise CameraNameError(3, "Переименование камеры доступно в Windows.")
            try:
                import winreg
            except ImportError as error:
                raise CameraNameError(3, "API реестра Windows недоступен.") from error

            module = winreg
        self.reg = module

    def flag(self, view):
        return self.reg.KEY_WOW64_64KEY if view == 64 else self.reg.KEY_WOW64_32KEY

    def read(self, root, path, name, view):
        with self.reg.OpenKey(root, path, 0, self.reg.KEY_QUERY_VALUE | self.flag(view)) as key:
            value, kind = self.reg.QueryValueEx(key, name)
            if kind != self.reg.REG_SZ or not isinstance(value, str):
                raise CameraNameError(3, "Неожиданный тип значения в регистрации камеры.")
            return value

    def snapshot(self):
        records = []
        for view in (32, 64):
            # A per-user COM override can hide the machine registration. Do not silently modify it.
            for path in (CLASS_KEY, INSTANCE_KEY):
                try:
                    with self.reg.OpenKey(
                        self.reg.HKEY_CURRENT_USER, path, 0, self.reg.KEY_QUERY_VALUE | self.flag(view)
                    ):
                        raise CameraNameError(
                            3,
                            "Найдена пользовательская регистрация камеры. "
                            "Автоматическое переименование поддерживает стандартную установку OBS.",
                        )
                except FileNotFoundError:
                    pass
            try:
                module = self.read(self.reg.HKEY_LOCAL_MACHINE, CLASS_KEY + r"\InprocServer32", "", view)
            except FileNotFoundError:
                continue
            try:
                records.append(
                    {
                        "view": view,
                        "module": module,
                        "friendly": self.read(
                            self.reg.HKEY_LOCAL_MACHINE, INSTANCE_KEY, "FriendlyName", view
                        ),
                        "class_name": self.read(self.reg.HKEY_LOCAL_MACHINE, CLASS_KEY, "", view),
                    }
                )
            except FileNotFoundError as error:
                raise CameraNameError(
                    3, "Регистрация камеры OBS неполная. Восстановите установку OBS."
                ) from error
        if not records:
            raise CameraNameError(3, "OBS Virtual Camera не установлена. Сначала установите OBS для Windows.")
        return registrations(records)

    @contextmanager
    def updates(self, views):
        handles = {}
        reg = self.reg
        with ExitStack() as stack:
            # Open every required key before changing any value. Missing permissions cause no partial write.
            for view in views:
                for field, (path, name) in FIELDS.items():
                    handles[view, field] = (
                        stack.enter_context(
                            reg.OpenKey(
                                reg.HKEY_LOCAL_MACHINE,
                                path,
                                0,
                                reg.KEY_QUERY_VALUE | reg.KEY_SET_VALUE | self.flag(view),
                            )
                        ),
                        name,
                    )

            class Writer:
                def get(self, view, field):
                    key, name = handles[view, field]
                    value, kind = reg.QueryValueEx(key, name)
                    if kind != reg.REG_SZ:
                        raise CameraNameError(3)
                    return value

                def set(self, view, field, value):
                    key, name = handles[view, field]
                    reg.SetValueEx(key, name, 0, reg.REG_SZ, value)

            yield Writer()


def target_state(expected, targets):
    if not isinstance(targets, dict) or not targets or not set(targets) <= {str(r.view) for r in expected}:
        raise CameraNameError(4)
    result = []
    for entry in expected:
        values = targets.get(str(entry.view))
        if values is None:
            result.append(entry)
            continue
        if not isinstance(values, dict) or set(values) != set(FIELDS):
            raise CameraNameError(4)
        result.append(replace(entry, **{field: valid_name(value, 512) for field, value in values.items()}))
    return tuple(result)


def change_names(registry, expected, targets):
    desired = target_state(expected, targets)
    if registry.snapshot() != expected:
        raise CameraNameError(2)
    if desired == expected:
        return
    before = {r.view: r for r in expected}
    wanted = {r.view: r for r in desired}
    with registry.updates(sorted(int(view) for view in targets)) as writer:
        if registry.snapshot() != expected:
            raise CameraNameError(2)
        undo = []
        try:
            for view in sorted(int(v) for v in targets):
                for field in FIELDS:
                    value, target = writer.get(view, field), getattr(wanted[view], field)
                    if value == target:
                        continue  # 32/64-bit views may share a key.
                    if value != getattr(before[view], field):
                        raise CameraNameError(2)
                    undo.append((view, field, value, target))
                    writer.set(view, field, target)
            if registry.snapshot() != desired:
                raise CameraNameError(2)
        except Exception as error:
            rollback_ok = True
            for view, field, old, target in reversed(undo):
                try:
                    current = writer.get(view, field)
                    if current == target:
                        writer.set(view, field, old)
                    elif current != old:
                        rollback_ok = False  # Preserve another process's concurrent edit.
                except Exception:
                    rollback_ok = False
            try:
                rollback_ok &= registry.snapshot() == expected
            except Exception:
                rollback_ok = False
            raise CameraNameError(6 if rollback_ok else 7) from error


def load_original(path):
    path = Path(path)
    if not path.exists():
        return ()
    if path.stat().st_size > 32768:
        raise CameraNameError(4, "Копия исходного имени повреждена; изменения не выполняются.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("format") != "framepilot-camera-names"
            or data.get("version") != 1
        ):
            raise CameraNameError(4)
        return registrations(data["original"])
    except (ValueError, KeyError, TypeError, CameraNameError) as error:
        raise CameraNameError(
            4, "Не удалось прочитать копию исходного имени; она не будет перезаписана."
        ) from error


def remember_original(path, current):
    from PySide6.QtCore import QLockFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(path) + ".lock")
    if not lock.tryLock(0):
        raise CameraNameError(4, "Другая копия FramePilot сохраняет исходное имя. Повторите действие.")
    try:
        _remember_original_locked(path, current)
    finally:
        lock.unlock()


def _remember_original_locked(path, current):
    saved = {r.view: r for r in load_original(path)}
    for item in current:
        saved.setdefault(item.view, item)
    data = {
        "format": "framepilot-camera-names",
        "version": 1,
        "original": [asdict(saved[view]) for view in sorted(saved)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".camera-names-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def rename_targets(current, name):
    name = valid_name(name)
    return {str(r.view): {"friendly": name, "class_name": name} for r in current}


def restore_targets(current, original):
    saved = {r.view: r for r in original}
    targets = {
        str(r.view): {field: getattr(saved[r.view], field) for field in FIELDS}
        for r in current
        if r.view in saved
    }
    if not targets:
        raise CameraNameError(4, "Нет сохранённого исходного имени для установленной камеры.")
    return targets


def task_token(current, targets):
    target_state(current, targets)
    body = json.dumps(
        {"version": 1, "expected": [asdict(r) for r in current], "targets": targets},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    token = base64.urlsafe_b64encode(body).decode()
    if len(token) > 16000:
        raise CameraNameError(4)
    return token


def run_task(token):
    """Elevated helper: accepts only names and expected state; writes no files or arbitrary keys."""
    try:
        if sys.platform != "win32":
            return 3
        from .windows_elevation import is_admin

        if not is_admin():
            return 5
        if not isinstance(token, str) or len(token) > 16000:
            return 4
        data = json.loads(base64.b64decode(token, altchars=b"-_", validate=True))
        if set(data) != {"version", "expected", "targets"} or data["version"] != 1:
            return 4
        current = registrations(data["expected"])
        change_names(WindowsRegistry(), current, data["targets"])
        return 0
    except CameraNameError as error:
        return error.code
    except PermissionError:
        return 5
    except Exception:
        return 4
