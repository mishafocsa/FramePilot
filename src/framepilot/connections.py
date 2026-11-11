"""Connection wizard. Observed, user-confirmed and untested stages remain distinct."""

import json
import os
import plistlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .camera import CameraInput
from .camera_identity import camera_label, is_obs_camera, registered_names, remember_obs_devices
from .obs import websocket_config_path
from .probe import ProbeHistory, VideoProbe
from .widgets import ImageCanvas, to_array, to_image

DESTINATIONS = {
    "genymotion": (
        "Genymotion Desktop",
        "Genymotion Desktop 3.10+ поддерживает OBS Virtual Camera экспериментально. "
        "В Media Injection выберите OBS Virtual Camera для Front/Back. "
        "По документации Genymotion нужен выход OBS 1280×720.",
    ),
    "genymotion_cloud": (
        "Genymotion SaaS в браузере",
        "Запустите виртуальную камеру OBS. В Media Injection нажмите "
        "GRANT ACCESS, разрешите браузеру камеру и выберите OBS Virtual Camera.",
    ),
    "bluestacks": (
        "BlueStacks Air",
        "Совместимость OBS Virtual Camera с выбранной версией Air ещё нужно проверить. "
        "Если виртуальную камеру нельзя выбрать, мастер не сможет добавить эту возможность эмулятору.",
    ),
    "online": (
        "Другой онлайн-сервис",
        "Сервис должен поддерживать передачу камеры браузера внутрь Android "
        "или загрузку видео как вход камеры. Обычная загрузка файла в галерею этого не подтверждает.",
    ),
    "other": (
        "Другое приложение",
        "Выберите OBS Virtual Camera в приложении и откройте его предпросмотр камеры.",
    ),
}
GENYMOTION_GUIDE = "https://support.genymotion.com/hc/en-us/articles/36992223861021-How-to-use-OBS-Virtual-Camera-with-Genymotion"


def safe_service_url(value):
    parts = urlsplit(value.strip())
    if parts.scheme not in ("https", "http") or not parts.hostname or parts.username or parts.password:
        raise ValueError("Укажите адрес сервиса http:// или https:// без логина и пароля")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def environment_report():
    roots = [Path("/Applications"), Path.home() / "Applications"]
    names = {
        "obs": ["OBS.app"],
        "genymotion": ["Genymotion.app"],
        "bluestacks": ["BlueStacks.app", "BlueStacks Air.app"],
        "android_studio": ["Android Studio.app"],
    }
    if sys.platform == "win32":
        roots = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
        ]
        names = {
            "obs": ["obs-studio/bin/64bit/obs64.exe"],
            "genymotion": ["Genymobile/Genymotion/genymotion.exe"],
            "bluestacks": ["BlueStacks_nxt/HD-Player.exe"],
            "android_studio": ["Android/Android Studio/bin/studio64.exe"],
        }
    result = {}
    for key, candidates in names.items():
        path = next((root / name for root in roots for name in candidates if (root / name).exists()), None)
        version = None
        if path and sys.platform == "darwin":
            try:
                version = plistlib.loads((path / "Contents/Info.plist").read_bytes()).get(
                    "CFBundleShortVersionString"
                )
            except (OSError, ValueError, TypeError):
                pass
        result[key] = {"installed": path is not None, "version": version}
    config = websocket_config_path()
    try:
        data = json.loads(config.read_text())
        result["obs_websocket"] = {
            "enabled_on_disk": bool(data.get("server_enabled")),
            "port_on_disk": data.get("server_port", 4455),
        }
    except (OSError, ValueError, TypeError):
        result["obs_websocket"] = {"enabled_on_disk": None, "port_on_disk": 4455}
    return result


class ProbeWorker(QObject):
    completed = Signal(int, object, object)
    failed = Signal(int, str, object)

    def __init__(self):
        super().__init__()
        self.capture = None

    @Slot(object)
    def observe(self, job):
        generation, probe, image, visible, window, observed_at = job
        try:
            started = time.monotonic()
            if image is None:
                if self.capture is None:
                    from .capture import create_capture

                    self.capture = create_capture()
                image = self.capture.capture(window)
                observed_at = time.monotonic()
            observation = probe.observe(
                image,
                visible,
                observed_at=observed_at,
                capture_started=started if window is not None else observed_at,
            )
            self.completed.emit(generation, observation, image)
        except Exception as error:
            self.failed.emit(generation, str(error), image)

    @Slot()
    def close(self):
        if self.capture is not None and hasattr(self.capture, "close"):
            self.capture.close()


class ConnectionWizard(QDialog):
    analyze = Signal(object)

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        main.connection_wizard = self
        self.setWindowTitle("FramePilot · мастер подключения и проверки")
        self.resize(1040, 820)
        self.environment = environment_report()
        self.probe = None
        self.history = ProbeHistory()
        self.active = self.busy = self.closing = False
        self.generation = 0
        self.visible_roi = None
        self.last_error = ""
        self.obs_details = None
        self.calibration_applied = False
        self.signals_detached = False
        self.local_stats = main.telemetry.summary()
        self.return_callbacks = 0
        self.return_first = self.return_last = None
        self.return_scope = None
        self.camera = CameraInput(self)
        self.camera.frame.connect(self.camera_frame)
        self.camera.message.connect(self.camera_message)
        self.thread = QThread(self)
        self.worker = ProbeWorker()
        self.worker.moveToThread(self.thread)
        self.analyze.connect(self.worker.observe)
        self.worker.completed.connect(self.observed)
        self.worker.failed.connect(self.failed)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.worker.close)
        self.thread.start()
        self.output_timer = QTimer(self)
        self.output_timer.timeout.connect(self.emit_pattern)
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_window)
        self.capture_timer.setInterval(200)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(500)
        self.build_ui()
        self.restore_connection()
        main.capture_worker.windows_ready.connect(self.refresh_windows)
        main.obs_worker.completed.connect(self.obs_completed)
        main.obs_worker.failed.connect(self.obs_failed)

    def label(self, text, layout):
        widget = QLabel(text)
        widget.setTextFormat(Qt.TextFormat.PlainText)
        widget.setWordWrap(True)
        layout.addWidget(widget)
        return widget

    def button(self, text, callback, layout):
        widget = QPushButton(text)
        widget.clicked.connect(callback)
        layout.addWidget(widget)
        return widget

    def build_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        def page(title):
            widget = QWidget()
            content = QVBoxLayout(widget)
            self.tabs.addTab(widget, title)
            return content

        target = page("1 · Назначение")
        self.destination = QComboBox()
        for key, (name, _) in DESTINATIONS.items():
            self.destination.addItem(name, key)
        target.addWidget(self.destination)
        self.destination_help = self.label("", target)
        self.destination.currentIndexChanged.connect(self.destination_changed)
        self.destination_changed()
        self.service_url = QLineEdit()
        self.service_url.setPlaceholderText("Адрес вашего онлайн-сервиса (необязательно)")
        target.addWidget(self.service_url)
        self.button("Открыть указанный сервис", self.open_service, target)
        self.button(
            "Официальная инструкция Genymotion / OBS",
            lambda: QDesktopServices.openUrl(QUrl(GENYMOTION_GUIDE)),
            target,
        )
        installed = [
            name
            for key, (name, _) in DESTINATIONS.items()
            if key in self.environment and self.environment[key].get("installed")
        ]
        self.label(
            "На этом компьютере: OBS "
            + str(self.environment["obs"]["version"] or "не найден")
            + "; эмуляторы: "
            + (", ".join(installed) if installed else "не найдены в стандартных папках"),
            target,
        )
        self.label(
            "Измерение в этой программе не подтверждает совместимость всех Android-приложений. "
            "Для онлайн-сервиса выходное видео передаётся в облако, когда вы включаете его камеру.",
            target,
        )
        target.addStretch()

        output = page("2 · Выход")
        self.label(
            "Проверьте исходный ролик в главном окне. Ниже показана локальная обработка до приёмника.", output
        )
        self.local_status = self.label("", output)
        self.button("Применить выход 720p / 30 FPS", self.apply_720p, output)
        self.button("Включить локальный видеовыход", self.main.start_local_output, output)
        self.output_status = self.label("", output)
        self.button("Начать проверочную сетку", self.start_probe, output)
        self.button("Остановить сетку и вернуться к видео", self.stop_probe, output)
        self.label(
            "Сетка содержит метки текущего сеанса и меняющийся счётчик. Старый кадр другой проверки "
            "не будет считаться подтверждением. Исходный ролик/камера при проверке приостанавливается.",
            output,
        )
        output.addStretch()

        obs = page("3 · OBS")
        self.label(
            "В OBS: Сервис → Настройки сервера WebSocket → включить сервер. "
            "Введите порт и пароль. Пароль не попадает в отчёт или профиль.",
            obs,
        )
        form = QFormLayout()
        self.obs_port = QSpinBox()
        self.obs_port.setRange(1, 65535)
        self.obs_port.setValue(self.environment.get("obs_websocket", {}).get("port_on_disk") or 4455)
        self.obs_password = QLineEdit()
        self.obs_password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Порт", self.obs_port)
        form.addRow("Пароль", self.obs_password)
        obs.addLayout(form)
        self.button("Подготовить сцену FramePilot", lambda: self.connect_obs(False), obs)
        self.button("Переключить OBS на FramePilot и включить камеру", lambda: self.connect_obs(True), obs)
        self.button("Проверить параметры OBS", self.diagnose_obs, obs)
        rename = self.button("Имя виртуальной камеры…", self.main.configure_camera_name, obs)
        rename.setVisible(sys.platform == "win32")
        self.obs_status = self.label("OBS ещё не проверен", obs)
        self.label(
            "В настройках OBS Virtual Camera выберите Program или сцену FramePilot. "
            "Проверка приёмника ниже покажет, действительно ли дошла наша сетка. "
            "Мастер не переключает сцену во время записи/трансляции.",
            obs,
        )
        obs.addStretch()

        receiving = page("4 · Приёмник")
        self.receiver = QComboBox()
        self.receiver.addItem("Окно приложения / браузера", "window")
        self.receiver.addItem("OBS Virtual Camera на этом компьютере", "camera")
        self.receiver.currentIndexChanged.connect(self.selection_changed)
        receiving.addWidget(self.receiver)
        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self.selection_changed)
        receiving.addWidget(self.window_combo)
        self.button("Обновить окна (нужно разрешение записи экрана)", self.main.refresh_windows, receiving)
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self.selection_changed)
        receiving.addWidget(self.device_combo)
        self.button("Обновить список OBS-камер", self.refresh_cameras, receiving)
        self.button("Начать проверку приёмника", self.start_probe, receiving)
        self.return_view = ImageCanvas("Покажите камеру в приёмнике; выделите её видимую область", drag=True)
        self.return_view.setMinimumHeight(220)
        self.return_view.selected.connect(self.set_visible_roi)
        receiving.addWidget(self.return_view, 1)
        self.button("Использовать всё изображение", self.clear_visible_roi, receiving)
        self.return_status = self.label("Проверка не запускалась", receiving)
        self.android_confirmation = QCheckBox(
            "Подтверждаю: выбранное окно показывает камеру Android-приложения"
        )
        receiving.addWidget(self.android_confirmation)
        self.apply_calibration_button = self.button(
            "Применить измеренные обрезку, зеркало и поворот", self.apply_calibration, receiving
        )
        self.apply_calibration_button.setEnabled(False)
        self.label(
            "Выделите именно видимую область камеры. Замер задержки включает наблюдение/захват "
            "и не является временем от сенсора до дисплея. Частота снимков окна — максимум 5 Гц, "
            "по ней нельзя определить FPS Android-камеры. Применение калибровки доступно "
            "в режиме «Камера / видеослот»; для полного холста согласуйте его с принимающей поверхностью.",
            receiving,
        )
        self.refresh_windows()

        summary = page("5 · Отчёт")
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        summary.addWidget(self.report_view, 1)
        self.button("Сохранить отчёт JSON…", self.save_report, summary)
        self.button("Запомнить подключение и параметры видео", self.save_connection, summary)
        self.button("Восстановить сохранённые параметры видео", self.restore_video_profile, summary)
        self.label(
            "Отчёт содержит числовые результаты и области измерения. Кадры, исходное видео "
            "и пароль OBS в отчёт не включаются.",
            summary,
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.selection_changed()

    def destination_changed(self):
        self.destination_help.setText(DESTINATIONS[self.destination.currentData()][1])

    def open_service(self):
        try:
            QDesktopServices.openUrl(QUrl(safe_service_url(self.service_url.text())))
        except ValueError as error:
            self.destination_help.setText(str(error))

    def apply_720p(self):
        self.stop_probe()
        self.main.auto_aspect.setChecked(False)
        self.main.resolution.setCurrentIndex(0)
        self.main.fps_combo.setCurrentIndex(0)
        self.main.read_settings()

    def refresh_windows(self, *_):
        self.stop_probe()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for index in range(self.main.window_combo.count()):
            self.window_combo.addItem(
                self.main.window_combo.itemText(index), self.main.window_combo.itemData(index)
            )
        self.window_combo.setCurrentIndex(self.main.window_combo.currentIndex())
        self.window_combo.blockSignals(False)
        self.selection_changed()

    def refresh_cameras(self):
        self.stop_probe()
        entries = registered_names()
        devices = CameraInput.devices()
        remember_obs_devices(devices, entries)
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for device in devices:
            if is_obs_camera(device, entries):
                self.device_combo.addItem(camera_label(device, entries), device)
        self.device_combo.blockSignals(False)
        self.selection_changed()
        if self.device_combo.count() == 0:
            self.return_status.setText(
                "OBS Virtual Camera не найдена. Включите её в OBS и проверьте установку виртуальной камеры."
            )

    def selection_changed(self, *_):
        self.stop_probe()
        self.history = ProbeHistory()
        self.return_scope = None
        self.last_error = ""
        self.calibration_applied = False
        self.apply_calibration_button.setEnabled(False)
        self.visible_roi = None
        self.android_confirmation.setChecked(False)
        self.android_confirmation.setEnabled(
            self.receiver.currentData() == "window" and self.window_combo.currentData() is not None
        )
        self.return_view.boxes = []
        self.return_view.update()
        if self.receiver.currentData() == "window":
            self.main.window_combo.setCurrentIndex(self.window_combo.currentIndex())

    def set_visible_roi(self, rect):
        self.generation += 1
        self.visible_roi = rect
        self.history = ProbeHistory()
        self.apply_calibration_button.setEnabled(False)
        self.return_view.boxes = [rect]
        self.return_view.update()

    def clear_visible_roi(self):
        self.generation += 1
        self.visible_roi = None
        self.history = ProbeHistory()
        self.apply_calibration_button.setEnabled(False)
        self.return_view.boxes = []
        self.return_view.update()

    def start_probe(self):
        if self.active:
            return
        if self.main.emergency_black:
            self.return_status.setText("Выход закрыт кнопкой «Чёрный кадр». Сначала нажмите «Вернуть видео».")
            return
        if self.receiver.currentData() == "camera" and self.device_combo.currentData() is None:
            self.refresh_cameras()
            if self.device_combo.currentData() is None:
                return
        self.main.start_local_output()
        if not self.main.publisher.url:
            self.return_status.setText("Сначала включите локальный выход")
            return
        self.local_stats = self.main.telemetry.summary()
        self.was_playing = self.main.player.playbackState() == self.main.player.PlaybackState.PlayingState
        self.was_camera_playing = self.main.active_source == "camera" and not self.main.camera_input.paused
        self.was_screen_playing = self.main.active_source == "screen" and self.main.studio.screen.running
        self.old_output = QImage(self.main.last_output)
        self.main.pause_source()
        self.main.scan_timer.stop()
        self.main.animation_timer.stop()
        self.main.generation += 1
        self.main.invalidate_frame(clear=True)
        self.probe = VideoProbe(self.main.settings.width, self.main.settings.height)
        self.history = ProbeHistory()
        self.last_error = ""
        self.generation += 1
        self.active = True
        self.main.connection_probe = self.probe
        self.main.watch_signal()
        self.calibration_applied = False
        self.return_callbacks = 0
        self.return_first = self.return_last = None
        window = self.window_combo.currentData()
        self.return_scope = (
            "local_obs_virtual_camera"
            if self.receiver.currentData() == "camera"
            else "framepilot_internal_demo"
            if window is None
            else "selected_external_window"
        )
        self.emit_pattern()
        self.output_timer.start(round(1000 / [30, 24, 15][self.main.fps_combo.currentIndex()]))
        if self.receiver.currentData() == "camera":
            self.camera.start(self.device_combo.currentData())
        else:
            if window is None:
                self.main.demo.show()
            self.capture_timer.start()
        self.return_status.setText("Сетка отправляется. Ожидание приёмника…")

    def emit_pattern(self):
        if not self.active:
            return
        image, _ = self.probe.frame()
        self.main.show_processed(image)
        self.main.publisher.publish(image, self.main.session)

    def capture_window(self):
        if not self.active or self.busy:
            return
        window = self.window_combo.currentData()
        image = to_array(self.main.demo.grab().toImage()) if window is None else None
        self.busy = True
        self.analyze.emit((self.generation, self.probe, image, self.visible_roi, window, time.monotonic()))

    def camera_frame(self, frame):
        if not self.active or not frame.isValid():
            return
        now = time.monotonic()
        self.return_callbacks += 1
        if self.return_first is None:
            self.return_first = now
        self.return_last = now
        if self.busy:
            return
        image = frame.toImage()
        if image.isNull():
            return
        self.busy = True
        self.analyze.emit((self.generation, self.probe, to_array(image), self.visible_roi, None, now))

    def camera_message(self, message):
        if not self.closing:
            self.return_status.setText(message)

    def observed(self, generation, observation, image):
        self.busy = False
        if generation != self.generation or not self.active or self.closing:
            return
        self.last_error = ""
        self.history.add(observation)
        self.return_view.show_image(to_image(image))
        delay = (
            f"{observation.observed_delay_ms:.0f} мс"
            if observation.observed_delay_ms is not None
            else "счётчик не виден"
        )
        self.return_status.setText(
            f"Найдена сетка текущего сеанса · {observation.markers} меток · "
            f"задержка наблюдения: {delay} · зеркало: {'да' if observation.mirror else 'нет'} · "
            f"поворот: {observation.rotation}°"
        )
        can_measure_crop = self.receiver.currentData() == "camera" or self.visible_roi is not None
        self.apply_calibration_button.setEnabled(
            can_measure_crop
            and self.main.settings.mapping == "camera"
            and self.history.stable_calibration() is not None
        )
        if not can_measure_crop:
            self.return_status.setText(
                self.return_status.text() + " · выделите видимую область камеры для измерения обрезки"
            )

    def failed(self, generation, message, image=None):
        self.busy = False
        if generation != self.generation or not self.active or self.closing:
            return
        self.last_error = message
        if image is not None:
            self.return_view.show_image(to_image(image))
        self.return_status.setText(message)
        self.apply_calibration_button.setEnabled(False)

    def stop_probe(self):
        if not self.active:
            return
        self.active = False
        self.generation += 1
        self.output_timer.stop()
        self.capture_timer.stop()
        self.camera.stop()
        self.main.connection_probe = None
        self.main.invalidate_frame(clear=True)
        if not self.old_output.isNull():
            self.main.show_processed(to_array(self.old_output))
        else:
            import numpy as np

            self.main.show_processed(
                np.zeros((self.main.settings.height, self.main.settings.width, 3), np.uint8)
            )
        self.main.publisher.publish(to_array(self.main.last_output), self.main.session)
        self.main.request_render()
        self.main.scan_timer.start(self.main.scan_interval())
        if self.was_playing and self.main.active_source == "file":
            self.main.player.play()
        elif (
            self.was_camera_playing and self.main.active_source == "camera" and self.main.camera_input.paused
        ):
            self.main.camera_input.toggle_pause()
        elif (
            self.was_screen_playing and self.main.active_source == "screen" and self.main.studio.screen.paused
        ):
            self.main.studio.screen.toggle_pause()

    def connect_obs(self, activate):
        if self.main.obs_busy:
            self.obs_status.setText("Дождитесь завершения предыдущего запроса OBS")
            return
        self.main.connect_obs(self.obs_port.value(), self.obs_password.text(), activate)
        self.obs_password.clear()

    def diagnose_obs(self):
        if self.main.obs_connected and not self.main.obs_busy:
            self.main.obs_busy = True
            self.main.jobs.obs.emit(("diagnose", ()))
        else:
            self.obs_status.setText("Сначала подключитесь к OBS")

    def obs_completed(self, state):
        if self.closing:
            return
        if state.get("diagnostics"):
            self.obs_details = state["diagnostics"]
            video = self.obs_details["video"]
            resolution = f"{video.get('outputWidth')}×{video.get('outputHeight')}"
            self.obs_status.setText(
                f"OBS {self.obs_details.get('obs_version')} · выход {resolution} · "
                f"виртуальная камера {'включена' if state.get('virtual_camera') else 'выключена'}"
            )
            if self.destination.currentData().startswith("genymotion") and resolution != "1280×720":
                self.obs_status.setText(
                    self.obs_status.text() + "\nДля Genymotion задайте в OBS выход 1280×720."
                )
        elif state.get("connected"):
            self.obs_status.setText("OBS подключён. Нажмите «Проверить параметры OBS».")

    def obs_failed(self, message):
        if not self.closing:
            self.obs_details = None
            self.obs_status.setText(message)

    def apply_calibration(self):
        calibration = self.history.stable_calibration()
        if (
            calibration is None
            or not self.apply_calibration_button.isEnabled()
            or self.main.settings.mapping != "camera"
        ):
            return
        self.stop_probe()
        from dataclasses import replace

        self.main.settings = replace(self.main.settings, calibration=calibration)
        self.main.read_settings()
        self.main.update_output_preview()
        self.calibration_applied = True
        self.return_status.setText("Измеренная калибровка применена. Проверьте исходное видео в приёмнике.")

    def save_connection(self):
        self.stop_probe()
        service = ""
        if self.service_url.text().strip():
            try:
                service = safe_service_url(self.service_url.text())
            except ValueError as error:
                self.return_status.setText(str(error))
                return
        window = self.window_combo.currentData()
        self.main.connection_profile = {
            "destination": self.destination.currentData(),
            "service": service,
            "receiver": self.receiver.currentData(),
            "window": window.profile_key if window else "demo",
            "obs_port": self.obs_port.value(),
            "profile": self.main.capture_profile(),
        }
        if self.receiver.currentData() == "window":
            self.main.window_combo.setCurrentIndex(self.window_combo.currentIndex())
            self.main.remember_window_profile()
        self.main.save_preferences()
        self.return_status.setText(
            "Подключение сохранено без пароля. Проверка при следующем запуске выполняется заново."
        )

    def restore_connection(self):
        saved = self.main.connection_profile
        for combo, key in ((self.destination, "destination"), (self.receiver, "receiver")):
            index = combo.findData(saved.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        if isinstance(saved.get("service"), str) and saved["service"]:
            try:
                self.service_url.setText(safe_service_url(saved["service"]))
            except ValueError:
                pass
        port = saved.get("obs_port")
        if type(port) is int and 1 <= port <= 65535:
            self.obs_port.setValue(port)
        for index in range(self.window_combo.count()):
            window = self.window_combo.itemData(index)
            if (window.profile_key if window else "demo") == saved.get("window"):
                self.window_combo.setCurrentIndex(index)
                break

    def restore_video_profile(self):
        self.stop_probe()
        try:
            self.main.apply_profile(self.main.connection_profile.get("profile"))
            self.return_status.setText("Параметры видео восстановлены. Подключение ещё нужно проверить.")
        except (ValueError, TypeError):
            self.return_status.setText("Сохранённые параметры подключения отсутствуют или повреждены")

    def report(self):
        measured = len(self.history.rows) >= 2
        android = (
            measured
            and self.return_scope == "selected_external_window"
            and self.android_confirmation.isChecked()
        )
        fps = (
            (self.return_callbacks - 1) / (self.return_last - self.return_first)
            if self.return_last is not None
            and self.return_first is not None
            and self.return_last > self.return_first
            else None
        )
        service = ""
        if self.service_url.text().strip():
            try:
                service = safe_service_url(self.service_url.text())
            except ValueError:
                service = "invalid"
        return {
            "application": "FramePilot",
            "version": __version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "destination": self.destination.currentData(),
            "service": service,
            "environment": self.environment,
            "local_pipeline": self.local_stats,
            "signal_health": self.main.signal_report(),
            "stages": {
                "local_output": "running" if self.main.publisher.url else "not_running",
                "obs_api": "observed" if self.obs_details else "not_verified",
                "returned_pattern": "observed_current_session" if measured else "not_verified",
                "android_camera": "user_confirmed_in_selected_window" if android else "not_verified",
            },
            "obs": self.obs_details,
            "measurement_active": self.active,
            "return_scope": self.return_scope,
            "timing": self.history.summary(),
            "virtual_camera_callback_fps": round(fps, 2) if fps is not None else None,
            "fps_scope": "Callbacks delivered by local Qt camera; not Android FPS. Window sampling never measures camera FPS.",
            "last_observation": self.history.geometry[-1].record() if self.history.geometry else None,
            "calibration_applied": self.calibration_applied,
            "last_error": self.last_error,
            "limits": [
                "A matched local virtual camera does not verify an Android app.",
                "Window timing includes screenshot/copy and sampling delay.",
                "Real-video tracking accuracy needs visual/ground-truth review.",
            ],
        }

    def refresh_status(self):
        if self.closing:
            return
        if not self.active:
            self.local_stats = self.main.telemetry.summary()
        stats = self.local_stats
        self.local_status.setText(
            f"Обработано кадров: {stats['completed_frames']} · частота: {stats['recent_processing_fps'] or '—'} FPS\n"
            f"Обработка: {stats['mean_worker_ms'] or '—'} мс в среднем, p95 {stats['p95_worker_ms'] or '—'} мс\n"
            f"Пропущено при занятости: {stats['skipped_while_busy']}; по лимиту FPS: {stats['skipped_by_fps_limit']}\n"
            f"Цель не подтверждена: {stats['unconfirmed_target_frames']} кадров. Это не оценка точности распознавания."
        )
        publisher = self.main.publisher
        self.output_status.setText(
            f"Локальный выход: {publisher.url or 'выключен'} · кадров: {publisher.frames}"
        )
        self.report_view.setPlainText(json.dumps(self.report(), ensure_ascii=False, indent=2))

    def save_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт подключения", "FramePilot-connection-report.json", "JSON (*.json)"
        )
        if path:
            Path(path).write_text(json.dumps(self.report(), ensure_ascii=False, indent=2), encoding="utf-8")

    def done(self, result):
        if self.closing:
            super().done(result)
            return
        self.closing = True
        self.stop_probe()
        self.refresh_timer.stop()
        self.obs_password.clear()
        for signal, handler in (
            []
            if self.signals_detached
            else [
                (self.main.capture_worker.windows_ready, self.refresh_windows),
                (self.main.obs_worker.completed, self.obs_completed),
                (self.main.obs_worker.failed, self.obs_failed),
            ]
        ):
            signal.disconnect(handler)
        self.signals_detached = True
        self.thread.quit()
        if not self.thread.wait(6500):
            self.closing = False
            self.return_status.setText("Ожидание завершения захвата окна…")
            QTimer.singleShot(500, lambda: self.done(result))
            return
        if self.main.connection_wizard is self:
            self.main.connection_wizard = None
        super().done(result)
