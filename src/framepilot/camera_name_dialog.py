"""Windows display-name editor; UAC runs only after the user's explicit Apply/Restore action."""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from .camera import CameraInput
from .camera_identity import is_obs_camera, remember_obs_devices
from .windows_camera_names import (
    MAX_NAME,
    MESSAGES,
    CameraNameError,
    WindowsRegistry,
    load_original,
    remember_original,
    rename_targets,
    restore_targets,
    target_state,
    task_token,
    valid_name,
)
from .windows_elevation import run_elevated


class RenameJob(QThread):
    completed = Signal(object)

    def __init__(self, token, hwnd, runner, parent):
        super().__init__(parent)
        self.token, self.hwnd, self.runner = token, hwnd, runner

    def run(self):
        try:
            code = self.runner(self.token, self.hwnd)
            self.completed.emit((code, MESSAGES.get(code, "Windows не завершила изменение имени камеры.")))
        except CameraNameError as error:
            self.completed.emit((error.code, str(error)))
        except Exception:
            self.completed.emit((5, "Не удалось запустить изменение имени камеры."))


class CameraNameDialog(QDialog):
    def __init__(self, backup_path, parent=None, registry=None, runner=run_elevated):
        super().__init__(parent)
        self.registry = registry or WindowsRegistry()
        self.backup_path, self.runner = backup_path, runner
        self.current = self.expected_after = ()
        self.job = None
        self.restoring_name = False
        self.setWindowTitle("Имя виртуальной камеры · Windows")
        self.resize(550, 380)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Введите своё название установленной OBS Virtual Camera. Изменится имя в списках камер; "
            "передача видео продолжит работать через OBS."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.current_label = QLabel()
        self.current_label.setTextFormat(Qt.TextFormat.PlainText)
        self.current_label.setWordWrap(True)
        layout.addWidget(self.current_label)
        self.name = QLineEdit()
        self.name.setMaxLength(MAX_NAME)
        self.name.setPlaceholderText("Например: FramePilot Camera")
        self.name.setAccessibleName("Новое имя виртуальной камеры")
        self.name.textChanged.connect(self.validate)
        layout.addWidget(self.name)
        row = QHBoxLayout()
        self.apply_button = QPushButton("Переименовать")
        self.apply_button.clicked.connect(self.apply_name)
        row.addWidget(self.apply_button)
        self.restore_button = QPushButton("Вернуть исходное имя")
        self.restore_button.clicked.connect(self.restore_name)
        row.addWidget(self.restore_button)
        layout.addLayout(row)
        self.refresh_button = QPushButton("Обновить")
        self.refresh_button.clicked.connect(self.refresh)
        layout.addWidget(self.refresh_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        note = QLabel(
            "Windows запросит права администратора. Исходные имена сохраняются перед первым изменением. "
            "После изменения перезапустите принимающее приложение. Обновление OBS может сбросить имя."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.refresh()

    @property
    def busy(self):
        return self.job is not None

    def refresh(self):
        if self.busy:
            return
        try:
            self.current = self.registry.snapshot()
            self.original = load_original(self.backup_path)
            self.current_label.setText(
                "Текущее имя:\n" + "\n".join(f"{r.view}-бит: {r.friendly}" for r in self.current)
            )
            preferred = next((r for r in self.current if r.view == 64), self.current[0])
            self.name.setText(preferred.friendly)
            self.status.setText("Введите новое имя и нажмите «Переименовать».")
        except (CameraNameError, OSError) as error:
            self.current = self.original = ()
            self.current_label.setText("Камера недоступна для изменения имени")
            self.status.setText(str(error))
        self.validate()

    def validate(self):
        valid = False
        try:
            valid_name(self.name.text().strip())
            valid = True
        except CameraNameError:
            pass
        self.apply_button.setEnabled(bool(self.current) and valid and not self.busy)
        self.restore_button.setEnabled(
            bool(self.current) and bool(getattr(self, "original", ())) and not self.busy
        )
        self.name.setEnabled(not self.busy)
        self.refresh_button.setEnabled(not self.busy)

    def apply_name(self):
        if self.busy:
            return
        try:
            name = valid_name(self.name.text().strip())
            devices = CameraInput.devices()
            remember_obs_devices(devices, self.current)
            for device in devices:
                if device.description().strip().casefold() == name.casefold() and not is_obs_camera(
                    device, self.current
                ):
                    raise CameraNameError(
                        4, "Это имя уже использует другая камера. Выберите другое название."
                    )
            self.start_change(rename_targets(self.current, name), restore=False)
        except (CameraNameError, OSError) as error:
            self.status.setText(str(error))

    def restore_name(self):
        if self.busy:
            return
        try:
            self.start_change(restore_targets(self.current, load_original(self.backup_path)), restore=True)
        except (CameraNameError, OSError) as error:
            self.status.setText(str(error))

    def start_change(self, targets, restore):
        after = target_state(self.current, targets)
        if self.registry.snapshot() != self.current:
            raise CameraNameError(2)
        if after == self.current:
            self.status.setText("Камера уже использует это имя.")
            return
        remember_original(self.backup_path, self.current)
        self.original = load_original(self.backup_path)
        token = task_token(self.current, targets)
        self.expected_after = after
        self.restoring_name = restore
        self.job = RenameJob(token, int(self.winId()), self.runner, self)
        self.job.completed.connect(self.completed)
        self.job.finished.connect(self.finished_job)
        self.status.setText("Ожидание разрешения Windows и изменения имени…")
        self.validate()
        self.job.start()

    def completed(self, result):
        code, message = result
        try:
            current = self.registry.snapshot()
            if code == 0 and current != self.expected_after:
                code, message = (
                    2,
                    "Windows завершила операцию, но новое имя не подтверждено. Обновите список.",
                )
            self.current = current
            self.current_label.setText(
                "Текущее имя:\n" + "\n".join(f"{r.view}-бит: {r.friendly}" for r in current)
            )
            preferred = next((r for r in current if r.view == 64), current[0])
            self.name.setText(preferred.friendly)
        except (CameraNameError, OSError) as error:
            self.current = ()
            code = 2
            message = str(error)
        if code == 0 and self.restoring_name:
            message = "Исходное имя восстановлено. Перезапустите принимающие приложения."
        self.status.setText(message)

    def finished_job(self):
        job, self.job = self.job, None
        job.deleteLater()
        self.validate()

    def reject(self):
        if self.busy:
            self.status.setText("Дождитесь результата операции или отмените запрос прав в окне Windows.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.busy:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
