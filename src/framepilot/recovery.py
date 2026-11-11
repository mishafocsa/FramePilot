"""Atomic background autosave. Recovery never starts a camera or screen capture."""

import json
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .automation import Automation, source_identity
from .preferences import decode_selections, validate_profile


def write_recovery(path, snapshot):
    source = Path(snapshot["source"])
    identity = source_identity(source)
    if snapshot.get("expected_identity") is not None and snapshot["expected_identity"] != identity:
        raise ValueError("Ролик изменился; предыдущий сеанс сохранён")
    data = {
        **snapshot,
        "automation": snapshot["automation"].record(),
        "identity": identity,
        "format": "framepilot-recovery",
        "version": 1,
        "preview_override": {
            "time": snapshot["automation"].override_time,
            "slot": snapshot["automation"].override_slot,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".recovery-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_recovery(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Автосохранение слишком большое")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("format") != "framepilot-recovery"
        or data.get("version") != 1
        or not isinstance(data.get("source"), str)
        or type(data.get("position")) is not int
        or data["position"] < 0
        or data.get("source_kind", "file") not in ("file", "photo")
    ):
        raise ValueError("Некорректное автосохранение")
    data["profile"] = validate_profile(data.get("profile"))
    data["automation"] = Automation.from_record(data.get("automation"))
    override = data.get("preview_override", {})
    if not isinstance(override, dict) or not isinstance(override.get("slot", "single"), str):
        raise ValueError("Некорректное состояние редактирования")
    when = override.get("time")
    if when is not None and (type(when) not in (int, float) or not math.isfinite(when) or when < 0):
        raise ValueError("Некорректное время редактирования")
    data["automation"] = replace(
        data["automation"], override_time=when, override_slot=override.get("slot", "single")
    )
    decode_selections(data.get("selections", {}))
    if source_identity(data["source"]) != data.get("identity"):
        raise ValueError("Исходный ролик автосохранения изменён или недоступен")
    return data


class RecoveryWorker(QObject):
    loaded = Signal(object)
    written = Signal(object)

    @Slot(object)
    def load(self, path):
        try:
            self.loaded.emit((read_recovery(path), None))
        except Exception as error:
            self.loaded.emit((None, str(error)))

    @Slot(object)
    def write(self, job):
        path, snapshot = job
        try:
            write_recovery(path, snapshot)
            self.written.emit(None)
        except Exception as error:
            self.written.emit(str(error))

    @Slot()
    def finish(self):
        QThread.currentThread().quit()


class RecoveryStore(QObject):
    load_request = Signal(object)
    write_request = Signal(object)
    finish_request = Signal()
    available = Signal(object)
    status = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.thread = QThread(self)
        self.worker = RecoveryWorker()
        self.worker.moveToThread(self.thread)
        self.load_request.connect(self.worker.load)
        self.write_request.connect(self.worker.write)
        self.finish_request.connect(self.worker.finish)
        self.worker.loaded.connect(self.available.emit)
        self.worker.written.connect(self.done)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self.busy = False
        self.pending = None
        self.closing = False

    def save(self, snapshot):
        if self.closing or snapshot is None:
            return
        if self.busy:
            self.pending = snapshot
        else:
            self.busy = True
            self.write_request.emit((self.path, snapshot))

    def done(self, error):
        self.busy = False
        self.status.emit("Ошибка автосохранения: " + error if error else "Сеанс сохранён автоматически")
        if self.pending is not None and not self.closing:
            pending, self.pending = self.pending, None
            self.save(pending)

    def close(self, snapshot=None):
        if not self.closing:
            self.closing = True
            self.pending = None
            if snapshot is not None:
                self.write_request.emit((self.path, snapshot))
            self.finish_request.emit()
        return self.thread.wait(17000)
