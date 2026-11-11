import argparse
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QStandardPaths, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QImage, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .appearance import STYLE as STYLE
from .appearance import install_appearance
from .calibration import Calibration, calibration_card, received_preview
from .camera import CameraInput
from .camera_identity import camera_label, registered_names, remember_obs_devices
from .editor import EditorShell, ElidedLabel
from .exporter import ExportJob
from .framing import WHOLE, FramingRule, freeze_framing
from .geometry import Rect, StableTarget
from .masks import FrameMask
from .object_geometry import QuadSelection, inverse_region, valid_quad
from .output import LocalOutput
from .photos import PHOTO_EXTENSIONS, PhotoInput
from .pipeline import SceneSlot, Settings
from .preferences import Preferences, decode_selections, encode_selections, validate_profile
from .signal_health import SignalMonitor, SignalPolicy
from .studio import Studio
from .telemetry import FrameTelemetry
from .widgets import (
    CalibrationDialog,
    DemoTarget,
    ExportDialog,
    ImageCanvas,
    OBSSetupDialog,
    to_array,
    to_image,
)
from .workers import CaptureWorker, ExportWorker, OBSWorker, VideoWorker


class Jobs(QObject):
    video = Signal(object)
    drain_video = Signal(object)
    scan = Signal(object)
    list_windows = Signal()
    obs = Signal(object)
    export = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FramePilot 0.15.2 · автокадрирование")
        self.resize(1380, 900)
        self.setMinimumSize(1120, 740)
        self.settings = Settings()
        self.telemetry = FrameTelemetry()
        self.connection_probe = None
        self.connection_wizard = None
        self.signal_monitor = SignalMonitor()
        self.signal_state = {"state": "idle", "blocked": False}
        self.emergency_black = False
        self.protection_key = None
        self.protection_image = QImage()
        self.last_normal_output = QImage()
        directory = Path(
            os.environ.get("FRAMEPILOT_CONFIG_DIR")
            or QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        )
        self.preferences = Preferences(directory / "framepilot.json")
        self.saved_state = self.preferences.load()
        self.signal_policy = SignalPolicy.from_record(self.saved_state.get("signal_protection"))
        connection = self.saved_state.get("connection_profile", {})
        self.connection_profile = connection if isinstance(connection, dict) else {}
        profiles = self.saved_state.get("window_profiles", {})
        self.window_profiles = profiles if isinstance(profiles, dict) else {}
        self.object_seed = None
        self.object_seeds = {}
        self.scene_slots = []
        self.active_scene_slot = ""
        self.slot_stabilizers = {}
        self.last_tracks = {}
        self.last_tracking_quality = {}
        self.last_composition = {}
        self.last_framing = {}
        self.last_render_time = 0.0
        self.recovery_candidates = []
        self.active_source = "file"
        self.file_epoch = 0
        self.camera_epoch = 0.0
        self.pending_scenario = None
        self.pending_object_sample = False
        self.object_sample_active = False
        scenarios = self.saved_state.get("scenarios", {})
        self.scenarios = scenarios if isinstance(scenarios, dict) else {}
        self.video_path = None
        self.photo_path = None
        self.photo_ready = False
        self.photo_opacity = None
        self.last_output = QImage()
        self.obs_connected = False
        self.obs_busy = False
        self.obs_pending = None
        self.obs_last_synced = None
        self.export_busy = False
        self.export_result = None
        self.preferred_port = self.saved_state.get("output_port", 18765)
        if type(self.preferred_port) is not int or not 1 <= self.preferred_port <= 65535:
            self.preferred_port = 18765
        self.restoring = True
        self.roi = None
        self.last_directory = self.saved_state.get("last_directory", "")
        self.render_pending = False
        self.latest_media_time = 0.0
        self.seek_pending = False
        self.seek_target = 0.0
        self.jobs = Jobs()
        self.session = 0
        self.generation = 0
        self.busy = False
        self.scan_busy = False
        self.listing = False
        self.reset_pending = True
        self.click = None
        self.last_frame = 0.0
        self.metric_start = time.monotonic()
        self.processed = 0
        self.metrics_started = False
        self.stable = StableTarget()
        self.slots = []
        self.slot_image_size = (800, 500)
        self.manual_slot = False
        self.current_slot = 0
        self.latest_frame = None
        self.closing = False
        self.closing_requested = False
        self.sample_directory = None
        self.video_thread = QThread(self)
        self.capture_thread = QThread(self)
        self.obs_thread = QThread(self)
        self.export_thread = QThread(self)
        self.publisher = LocalOutput()
        self.video_worker = VideoWorker(self.publisher)
        self.capture_worker = CaptureWorker()
        self.obs_worker = OBSWorker()
        self.export_worker = ExportWorker()
        self.video_worker.moveToThread(self.video_thread)
        self.capture_worker.moveToThread(self.capture_thread)
        self.obs_worker.moveToThread(self.obs_thread)
        self.export_worker.moveToThread(self.export_thread)
        self.jobs.video.connect(self.video_worker.process)
        self.jobs.drain_video.connect(self.video_worker.drain)
        self.jobs.scan.connect(self.capture_worker.scan)
        self.jobs.list_windows.connect(self.capture_worker.list_windows)
        self.jobs.obs.connect(self.obs_worker.configure)
        self.jobs.export.connect(self.export_worker.export)
        self.video_worker.completed.connect(self.frame_ready)
        self.video_worker.failed.connect(self.video_error)
        self.capture_worker.scanned.connect(self.scan_ready)
        self.capture_worker.windows_ready.connect(self.windows_ready)
        self.capture_worker.failed.connect(self.capture_error)
        self.obs_worker.completed.connect(self.obs_finished)
        self.obs_worker.failed.connect(self.obs_failed)
        self.export_worker.progress.connect(lambda value: self.export_progress.setValue(value))
        self.export_worker.completed.connect(self.export_finished)
        self.export_worker.failed.connect(self.export_failed)
        self.video_thread.finished.connect(self.video_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_worker.close)
        self.obs_thread.finished.connect(self.obs_worker.deleteLater)
        self.obs_thread.finished.connect(self.obs_worker.disconnect)
        self.export_thread.finished.connect(self.export_worker.deleteLater)
        self.video_thread.start()
        self.capture_thread.start()
        self.obs_thread.start()
        self.export_thread.start()
        self.player = QMediaPlayer(self)
        self.camera_input = CameraInput(self)
        self.camera_input.frame.connect(lambda frame: self.video_frame(frame, source="camera"))
        self.camera_input.message.connect(self.camera_message)
        self.audio = QAudioOutput(self)
        self.audio.setMuted(True)
        self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink(self)
        self.player.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(
            lambda frame, epoch=self.file_epoch: self.file_video_frame(frame, epoch)
        )
        self.player.errorOccurred.connect(self.player_error)
        self.player.mediaStatusChanged.connect(self.media_status)
        self.output_window = ImageCanvas("Ожидание видео")
        self.output_window.setWindowTitle("FramePilot Output")
        self.output_window.resize(960, 540)
        self.demo = DemoTarget()
        self.photo_input = PhotoInput(self)
        self.photo_input.loaded.connect(self.photo_loaded)
        self.photo_input.failed.connect(self.photo_failed)
        self.studio = Studio(self)
        self.build_ui()
        self.restore_preferences()
        self.read_settings(persist=False)
        self.restoring = False
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_preferences)
        self.animation_timer = QTimer(self)
        self.animation_timer.setSingleShot(True)
        self.animation_timer.timeout.connect(self.request_render)
        self.setAcceptDrops(True)
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.open_video)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play)
        self.install_shortcuts()
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.scan_window)
        self.scan_timer.start(self.scan_interval())
        self.obs_timer = QTimer(self)
        self.obs_timer.timeout.connect(self.poll_obs)
        self.obs_timer.start(2000)
        self.obs_sync_timer = QTimer(self)
        self.obs_sync_timer.setSingleShot(True)
        self.obs_sync_timer.timeout.connect(self.flush_obs_sync)
        self.signal_timer = QTimer(self)
        self.signal_timer.timeout.connect(self.watch_signal)
        self.signal_timer.start(250)
        self.watch_signal()
        if self.saved_state.get("local_output_enabled") is True:
            QTimer.singleShot(0, self.start_local_output)

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        if isinstance(layout, QFormLayout):
            layout.addRow(button)
        else:
            layout.addWidget(button)
        return button

    def build_ui(self):
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Выберите камеру…", None)
        self.camera_combo.setMinimumContentsLength(10)
        self.camera_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.camera_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.loop = QCheckBox("Повтор")
        self.loop.setChecked(True)
        self.mute = QCheckBox("Без звука")
        self.mute.setChecked(True)
        self.mute.toggled.connect(self.audio.setMuted)
        self.filename = ElidedLabel("Добавьте видеофайл или включите камеру")
        self.filename.setObjectName("mutedText")
        self.position_label = QLabel("00:00 / 00:00")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.sliderReleased.connect(lambda: self.seek_to(self.timeline.value()))
        self.timeline.sliderMoved.connect(lambda value: self.update_position(value, label_only=True))
        self.timeline.actionTriggered.connect(lambda _: QTimer.singleShot(0, self.seek_from_keyboard))
        self.player.durationChanged.connect(self.duration_changed)
        self.player.positionChanged.connect(self.update_position)
        self.input_view = ImageCanvas("Исходное видео", drag=True)
        self.input_view.clicked.connect(self.select_face)
        self.input_view.selected.connect(self.select_input_region)
        self.output_view = ImageCanvas("Ваш готовый кадр")
        self.target_view = ImageCanvas("Рамка в приложении", drag=True)
        self.target_view.selected.connect(self.manual_frame)
        self.target_view.clicked.connect(self.choose_slot)
        self.status = ElidedLabel("Готово к загрузке видео")
        self.status.setObjectName("mutedText")
        self.metrics = ElidedLabel("Показатели появятся после запуска видео")
        self.metrics.setObjectName("performanceText")
        self.quality_label = ElidedLabel()
        self.quality_label.setVisible(False)
        self.signal_label = ElidedLabel("Сигнал: источник не выбран")
        self.recovery_button = QPushButton("Подтвердить найденную цель")
        self.recovery_button.setEnabled(False)
        self.recovery_button.clicked.connect(self.confirm_recovery)
        controls = QWidget()
        panel = QVBoxLayout(controls)
        group = QGroupBox("Рамка в приложении")
        form = QVBoxLayout(group)
        self.window_combo = QComboBox()
        self.window_combo.addItem("Тестовое приложение", None)
        self.window_combo.currentIndexChanged.connect(self.target_changed)
        form.addWidget(self.window_combo)
        row = QHBoxLayout()
        self.button("Тестовое окно", self.show_demo, row)
        self.refresh_button = self.button("Обновить окна", self.refresh_windows, row)
        form.addLayout(row)
        self.auto_slot = QCheckBox("Автоматически искать рамку")
        self.auto_slot.setChecked(True)
        self.auto_slot.toggled.connect(self.detection_changed)
        form.addWidget(self.auto_slot)
        self.sensitivity = QComboBox()
        self.sensitivity.addItem("Обычные рамки", "normal")
        self.sensitivity.addItem("Тонкие / слабоконтрастные", "sensitive")
        self.sensitivity.currentIndexChanged.connect(self.detection_changed)
        form.addWidget(self.sensitivity)
        roi_row = QHBoxLayout()
        self.roi_button = self.button("Область поиска", self.toggle_roi_selection, roi_row)
        self.roi_button.setCheckable(True)
        self.button("Всё окно", self.clear_roi, roi_row)
        form.addLayout(roi_row)
        row = QHBoxLayout()
        self.button("Другая рамка", self.next_slot, row)
        self.button("Автопоиск", self.resume_auto, row)
        form.addLayout(row)
        self.window_status = QLabel(
            "Для настоящего приложения нажмите «Обновить окна», разрешите запись экрана "
            "и выберите его в списке. Тестовое окно служит демонстрацией."
        )
        self.window_status.setWordWrap(True)
        form.addWidget(self.window_status)
        self.button("Разрешение записи экрана…", self.open_screen_permission_settings, form)
        panel.addWidget(group)

        group = QGroupBox("Рамки композиции · до 4")
        frame_layout = QVBoxLayout(group)
        self.scene_combo = QComboBox()
        self.scene_combo.addItem("Одна рамка", "")
        self.scene_combo.currentIndexChanged.connect(self.select_scene_slot)
        frame_layout.addWidget(self.scene_combo)
        row = QHBoxLayout()
        self.button("Добавить рамку", self.add_scene_slot, row)
        self.button("Удалить рамку", self.remove_scene_slot, row)
        frame_layout.addLayout(row)
        self.button("Рамки из предпросмотра", self.scene_from_detected, frame_layout)
        helper = QLabel(
            "Выберите рамку, затем укажите для неё лицо/человека кликом или предмет выделением. "
            "Результат — один общий видеокадр."
        )
        helper.setWordWrap(True)
        frame_layout.addWidget(helper)
        panel.addWidget(group)

        group = QGroupBox("Фон композиции")
        background_layout = QVBoxLayout(group)
        self.remove_background = QCheckBox("Удалить фон у людей")
        if sys.platform == "win32":
            self.remove_background.setEnabled(False)
            self.remove_background.setToolTip("Сегментация людей пока доступна только в macOS")
        self.remove_background.toggled.connect(self.read_settings)
        background_layout.addWidget(self.remove_background)
        self.segmentation_quality = QComboBox()
        self.segmentation_quality.addItem("Баланс скорости и качества", "balanced")
        self.segmentation_quality.addItem("Точная маска каждый кадр", "accurate")
        self.segmentation_quality.currentIndexChanged.connect(self.read_settings)
        background_layout.addWidget(self.segmentation_quality)
        row = QHBoxLayout()
        self.button("Цвет фона…", self.choose_background_color, row)
        self.button("Изображение…", self.choose_background_image, row)
        background_layout.addLayout(row)
        self.button("Убрать изображение фона", self.clear_background_image, background_layout)
        label = QLabel("Сегментация выделяет людей. У произвольных предметов исходный фон сохраняется.")
        label.setWordWrap(True)
        background_layout.addWidget(label)
        panel.addWidget(group)

        group = QGroupBox("Сценарии")
        scenario_layout = QVBoxLayout(group)
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("Выберите сценарий…", None)
        scenario_layout.addWidget(self.scenario_combo)
        row = QHBoxLayout()
        self.button("Сохранить…", self.save_scenario, row)
        self.button("Применить", self.apply_scenario, row)
        scenario_layout.addLayout(row)
        self.button("Удалить сценарий", self.remove_scenario, scenario_layout)
        label = QLabel(
            "Cmd+Option+1…4 — первые четыре сценария. Камера не запускается при открытии программы."
        )
        label.setWordWrap(True)
        scenario_layout.addWidget(label)
        panel.addWidget(group)
        self.refresh_scenarios()

        profile_group = QGroupBox("Профили окон")
        profile_layout = QVBoxLayout(profile_group)
        self.window_profile_combo = QComboBox()
        self.window_profile_combo.addItem("Профиль окна…", None)
        self.window_profile_combo.currentIndexChanged.connect(self.load_window_profile)
        profile_layout.addWidget(self.window_profile_combo)
        profile_buttons = QHBoxLayout()
        self.button("Запомнить для окна", self.remember_window_profile, profile_buttons)
        self.button("Удалить", self.remove_window_profile, profile_buttons)
        profile_layout.addLayout(profile_buttons)
        panel.addWidget(profile_group)
        self.refresh_window_profiles()

        group = QGroupBox("Цель и отслеживание")
        form = QFormLayout(group)
        self.subject_mode = QComboBox()
        self.subject_mode.addItem("Лицо (клик)", "face")
        self.subject_mode.addItem("Предмет (выделить)", "object")
        self.subject_mode.addItem("Человек по пояс", "upper")
        self.subject_mode.addItem("Человек целиком", "person")
        if sys.platform == "win32":
            for index in (2, 3):
                self.subject_mode.model().item(index).setEnabled(False)
        self.subject_mode.currentIndexChanged.connect(self.read_settings)
        form.addRow("Отслеживание", self.subject_mode)
        self.subject_region_combo = QComboBox()
        for label, region in [
            ("Вся цель", Rect(0, 0, 1, 1)),
            ("Левая половина", Rect(0, 0, 0.5, 1)),
            ("Правая половина", Rect(0.5, 0, 0.5, 1)),
            ("Верхняя половина", Rect(0, 0, 1, 0.5)),
            ("Нижняя половина", Rect(0, 0.5, 1, 0.5)),
            ("Произвольная часть", None),
        ]:
            self.subject_region_combo.addItem(label, region)
        self.subject_region_combo.currentIndexChanged.connect(self.subject_region_changed)
        form.addRow("Показывать", self.subject_region_combo)
        self.part_button = QPushButton("Выделить часть внутри цели")
        self.part_button.setCheckable(True)
        self.part_button.toggled.connect(self.begin_select_part)
        form.addRow(self.part_button)
        label = QLabel("Отслеживается вся цель. В выход попадает выбранная часть её прямоугольника.")
        label.setWordWrap(True)
        form.addRow(label)
        self.button("Выбрать цель заново", self.reset_tracking, form)
        form.addRow(self.recovery_button)
        panel.addWidget(group)
        group = QGroupBox("Автоматическая смена плана")
        form = QFormLayout(group)
        self.auto_framing_enabled = QCheckBox("Автопереход: часть ↔ целиком")
        self.auto_framing_enabled.toggled.connect(self.read_settings)
        form.addRow(self.auto_framing_enabled)
        self.whole_when = QComboBox()
        self.whole_when.addItem("Рамка больше порога", "larger")
        self.whole_when.addItem("Рамка меньше порога", "smaller")
        self.whole_when.currentIndexChanged.connect(self.read_settings)
        form.addRow("Показывать целиком", self.whole_when)
        self.frame_threshold = QDoubleSpinBox()
        self.frame_threshold.setRange(1, 95)
        self.frame_threshold.setValue(20)
        self.frame_threshold.setSuffix(" % окна")
        self.frame_threshold.valueChanged.connect(self.read_settings)
        form.addRow("Порог площади рамки", self.frame_threshold)
        self.frame_threshold_button = QPushButton("Текущая рамка — порог")
        self.frame_threshold_button.clicked.connect(self.remember_framing_threshold)
        form.addRow(self.frame_threshold_button)
        self.framing_hysteresis = QDoubleSpinBox()
        self.framing_hysteresis.setRange(0, 30)
        self.framing_hysteresis.setValue(10)
        self.framing_hysteresis.setSuffix(" % от порога")
        self.framing_hysteresis.valueChanged.connect(self.read_settings)
        form.addRow("Защита от переключений", self.framing_hysteresis)
        self.framing_transition = QDoubleSpinBox()
        self.framing_transition.setRange(0, 1)
        self.framing_transition.setDecimals(2)
        self.framing_transition.setSingleStep(0.02)
        self.framing_transition.setValue(0.12)
        self.framing_transition.setSuffix(" с")
        self.framing_transition.valueChanged.connect(self.read_settings)
        form.addRow("Переход часть / целиком", self.framing_transition)
        self.framing_status = QLabel("Выберите часть цели для автоматического переключения")
        self.framing_status.setWordWrap(True)
        form.addRow(self.framing_status)
        label = QLabel(
            "Порог относится к площади рамки в окне назначения. "
            "В режиме «целиком» цель вписывается без дополнительной обрезки."
        )
        label.setWordWrap(True)
        form.addRow(label)
        panel.addWidget(group)
        group = QGroupBox("Размещение")
        form = QFormLayout(group)
        self.mode = QComboBox()
        self.mode.addItem("Камера / видеослот", "camera")
        self.mode.addItem("Полный холст", "canvas")
        self.mode.currentIndexChanged.connect(self.read_settings)
        form.addRow("Размещение", self.mode)
        self.fit_combo = QComboBox()
        self.fit_combo.addItems(["Вписать целиком", "Заполнить рамку"])
        self.fit_combo.currentIndexChanged.connect(self.read_settings)
        form.addRow("Режим", self.fit_combo)
        self.padding = QDoubleSpinBox()
        self.padding.setRange(0, 1)
        self.padding.setSingleStep(0.05)
        self.padding.setValue(0.25)
        self.padding.valueChanged.connect(self.read_settings)
        form.addRow("Отступ вокруг цели", self.padding)
        self.motion_mode = QComboBox()
        self.motion_mode.addItem("Плавная", "smooth")
        self.motion_mode.addItem("Быстрая · минимальное сглаживание", "responsive")
        self.motion_mode.currentIndexChanged.connect(self.read_settings)
        form.addRow("Реакция", self.motion_mode)
        form.addRow("Масштаб кадра", self.studio.zoom_control())
        panel.addWidget(group)
        group = QGroupBox("Движение и ориентиры")
        form = QFormLayout(group)
        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.05, 2)
        self.smoothing.setSingleStep(0.05)
        self.smoothing.setValue(0.35)
        self.smoothing.setSuffix(" с")
        self.smoothing.valueChanged.connect(self.read_settings)
        form.addRow("Плавность", self.smoothing)
        self.align_eyes = QCheckBox("Удерживать глаза по высоте")
        self.align_eyes.toggled.connect(self.read_settings)
        form.addRow(self.align_eyes)
        self.eye_level = QDoubleSpinBox()
        self.eye_level.setRange(0.1, 0.8)
        self.eye_level.setSingleStep(0.02)
        self.eye_level.setValue(0.38)
        self.eye_level.valueChanged.connect(self.read_settings)
        form.addRow("Уровень глаз (0–1)", self.eye_level)
        self.max_zoom = QDoubleSpinBox()
        self.max_zoom.setRange(1, 12)
        self.max_zoom.setValue(6)
        self.max_zoom.valueChanged.connect(self.read_settings)
        form.addRow("Предел приближения, ×", self.max_zoom)
        self.loss = QComboBox()
        for name, value in [
            ("Удержать кадр", "freeze"),
            ("Исходный план", "original"),
            ("Скрыть цель", "black"),
        ]:
            self.loss.addItem(name, value)
        self.loss.currentIndexChanged.connect(self.read_settings)
        form.addRow("При потере цели", self.loss)
        panel.addWidget(group)

        group = QGroupBox("Калибровка принимающей программы")
        calibration_layout = QVBoxLayout(group)
        self.button("Обрезка, зеркало, положение…", self.configure_calibration, calibration_layout)
        self.receiver_preview = QCheckBox("Предпросмотр с учётом приёмника")
        self.receiver_preview.toggled.connect(self.update_output_preview)
        calibration_layout.addWidget(self.receiver_preview)
        self.grid = QCheckBox("Проверочная сетка вместо видео")
        self.grid.toggled.connect(self.toggle_grid)
        calibration_layout.addWidget(self.grid)
        panel.addWidget(group)

        group = QGroupBox("Вывод в OBS")
        form = QVBoxLayout(group)
        self.resolution = QComboBox()
        for label, value in [
            ("720p · 1280×720", (1280, 720)),
            ("1080p · 1920×1080", (1920, 1080)),
            ("Портрет · 720×1280", (720, 1280)),
        ]:
            self.resolution.addItem(label, value)
        self.resolution.currentIndexChanged.connect(self.read_settings)
        form.addWidget(self.resolution)
        self.auto_aspect = QCheckBox("Пропорции по найденной рамке")
        self.auto_aspect.setChecked(False)
        self.auto_aspect.toggled.connect(self.read_settings)
        form.addWidget(self.auto_aspect)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["30 FPS", "24 FPS", "15 FPS"])
        self.fps_combo.currentIndexChanged.connect(self.read_settings)
        form.addWidget(self.fps_combo)
        self.output_size = QLabel("1280×720")
        form.addWidget(self.output_size)
        self.button("Вывод в OBS через браузер", lambda: self.start_local_output(copy_address=True), form)
        self.local_url = QLineEdit()
        self.local_url.setReadOnly(True)
        self.local_url.setPlaceholderText("Локальный выход выключен")
        form.addWidget(self.local_url)
        row = QHBoxLayout()
        self.button("Открыть адрес", self.open_output_url, row)
        self.button("Остановить выход", self.stop_local_output, row)
        form.addLayout(row)
        self.obs_button = self.button("Настроить OBS автоматически…", self.configure_obs, form)
        self.camera_name_button = self.button("Имя виртуальной камеры…", self.configure_camera_name, form)
        self.camera_name_button.setVisible(sys.platform == "win32")
        self.obs_status = QLabel("OBS WebSocket: не подключён")
        self.obs_status.setWordWrap(True)
        form.addWidget(self.obs_status)
        self.button("Открыть чистое окно вывода", self.show_output, form)
        instructions = QLabel(
            "В OBS добавьте источник «Браузер» и вставьте локальный адрес. "
            "Установите размер источника как у выхода FramePilot.\n"
            "Затем включите OBS Virtual Camera и выберите её в принимающей программе. "
            "Альтернатива — захват чистого окна FramePilot Output."
        )
        instructions.setWordWrap(True)
        form.addWidget(instructions)
        panel.addWidget(group)
        group = QGroupBox("Если пропал видеосигнал")
        guard_form = QFormLayout(group)
        self.signal_timeout = QDoubleSpinBox()
        self.signal_timeout.setRange(0.5, 15)
        self.signal_timeout.setSingleStep(0.5)
        self.signal_timeout.setSuffix(" с")
        self.signal_timeout.setValue(self.signal_policy.timeout)
        guard_form.addRow("Считать сбоем через", self.signal_timeout)
        self.signal_fallback = QComboBox()
        self.signal_fallback.addItem("Чёрный кадр", "black")
        self.signal_fallback.addItem("Удержать последний кадр", "freeze")
        self.signal_fallback.setCurrentIndex(self.signal_fallback.findData(self.signal_policy.fallback))
        guard_form.addRow("При сбое источника", self.signal_fallback)
        self.signal_timeout.valueChanged.connect(self.signal_policy_changed)
        self.signal_fallback.currentIndexChanged.connect(self.signal_policy_changed)
        guard_help = QLabel(
            "Контроль входного видео и обработки. Обычная пауза не считается сбоем. "
            "При смене источника выход закрыт до первого свежего обработанного кадра. "
            "«Чёрный кадр» скрывает видео в выходе, сохраняя работу отслеживания. На экспорт MP4 не влияет."
        )
        guard_help.setWordWrap(True)
        guard_form.addRow(guard_help)
        panel.addWidget(group)
        group = QGroupBox("Экспорт видео")
        export_layout = QVBoxLayout(group)
        self.export_button = self.button("Сохранить обработанный MP4…", self.start_export, export_layout)
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        export_layout.addWidget(self.export_progress)
        self.export_cancel = self.button("Отменить экспорт", self.export_worker.cancel.set, export_layout)
        self.export_cancel.setEnabled(False)
        self.export_status = QLabel("H.264 MP4 · исходный звук по желанию")
        self.export_status.setWordWrap(True)
        export_layout.addWidget(self.export_status)
        panel.addWidget(group)
        profile_files = QGroupBox("Файлы профилей")
        row = QVBoxLayout(profile_files)
        self.button("Сохранить профиль", self.save_profile, row)
        self.button("Загрузить профиль", self.load_profile, row)
        panel.addWidget(profile_files)
        for group in self.studio.groups():
            panel.addWidget(group)
        panel.addStretch()
        for combo in controls.findChildren(QComboBox):
            combo.setMinimumContentsLength(10)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for form_layout in controls.findChildren(QFormLayout):
            form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        groups = []
        while panel.count():
            item = panel.takeAt(0)
            if isinstance(item.widget(), QGroupBox):
                groups.append(item.widget())
        self.editor = EditorShell(self, groups)
        controls.deleteLater()

    def read_settings(self, *_, persist=True):
        width, height = self.resolution.currentData()
        mapping = self.mode.currentData()
        if self.auto_aspect.isChecked() and mapping == "camera":
            r = self.settings.target.pixels(*self.slot_image_size)
            if r.w > 0 and r.h > 0:
                aspect = r.w / r.h
                edge = max(width, height)
                width, height = (edge, round(edge / aspect)) if aspect >= 1 else (round(edge * aspect), edge)
                width, height = max(64, width // 2 * 2), max(64, height // 2 * 2)
        previous = getattr(self, "render_settings", self.settings)
        self.settings = replace(
            self.settings,
            width=width,
            height=height,
            mapping=mapping,
            fill=self.fit_combo.currentIndex() == 1,
            padding=self.padding.value(),
            response=self.smoothing.value(),
            loss=self.loss.currentData(),
            subject_mode=self.subject_mode.currentData(),
            eye_level=self.eye_level.value() if self.align_eyes.isChecked() else None,
            max_zoom=self.max_zoom.value(),
            remove_background=self.remove_background.isChecked() and sys.platform == "darwin",
            segmentation_quality=self.segmentation_quality.currentData(),
            motion_mode=self.motion_mode.currentData(),
            frame_mask=self.studio.mask_value(),
            framing_zoom=self.studio.zoom.value(),
            surface_lock=self.studio.surface_lock.isChecked(),
            framing=FramingRule(
                self.auto_framing_enabled.isChecked(),
                self.whole_when.currentData(),
                self.frame_threshold.value() / 100,
                self.framing_hysteresis.value() / 100,
                self.framing_transition.value(),
            ),
        )
        if self.scene_slots:
            self.scene_slots = [
                replace(
                    slot,
                    target=self.settings.target,
                    shape=self.settings.shape,
                    subject_mode=self.settings.subject_mode,
                    padding=self.settings.padding,
                    fill=self.settings.fill,
                    eye_level=self.settings.eye_level,
                    max_zoom=self.settings.max_zoom,
                    subject_region=self.settings.subject_region,
                    framing=self.settings.framing,
                    frame_mask=self.settings.frame_mask,
                    framing_zoom=self.settings.framing_zoom,
                    surface_lock=self.settings.surface_lock,
                )
                if slot.identifier == self.active_scene_slot
                else slot
                for slot in self.scene_slots
            ]
            self.settings = replace(
                self.settings,
                slots=tuple(self.scene_slots),
                active_slot=self.active_scene_slot,
                mapping="canvas",
            )
        else:
            self.settings = replace(self.settings, slots=(), active_slot="")
        self.mode.setEnabled(not bool(self.scene_slots))
        if (
            not self.restoring
            and self.active_source == "file"
            and self.latest_frame is not None
            and any(
                getattr(previous, key) != getattr(self.settings, key)
                for key in ("padding", "subject_region", "framing_zoom")
            )
        ):
            self.settings = replace(
                self.settings,
                automation=replace(
                    self.settings.automation,
                    override_time=self.latest_media_time,
                    override_slot=self.active_scene_slot or "single",
                ),
            )
        self.studio.settings_changed()
        self.studio.surface_lock.setEnabled(self.settings.subject_mode == "object")
        self.target_view.boxes = (
            [slot.target for slot in self.scene_slots] if self.scene_slots else [self.settings.target]
        )
        self.target_view.update()
        self.smoothing.setEnabled(self.settings.motion_mode == "smooth")
        has_part = self.settings.subject_region != WHOLE
        for control in self.framing_controls():
            control.setEnabled(has_part and self.settings.framing.enabled)
        self.frame_threshold_button.setEnabled(has_part and self.settings.framing.enabled)
        if not has_part:
            self.framing_status.setText("Вся цель · выберите часть для автоматического переключения")
        elif not self.settings.framing.enabled:
            self.framing_status.setText("Фиксированная часть · автоматическое переключение выключено")
        else:
            self.framing_status.setText(
                f"Авто · площадь рамки {self.settings.target.area * 100:.1f}% окна · ожидание кадра"
            )
        if hasattr(self, "scan_timer"):
            self.scan_timer.setInterval(self.scan_interval())
        if previous.subject_mode != self.settings.subject_mode:
            if not self.scene_slots:
                self.reset_pending = True
            self.object_seed = None
        whole_target = self.settings.subject_region == Rect(0, 0, 1, 1)
        self.align_eyes.setEnabled(whole_target and self.settings.subject_mode == "face")
        self.eye_level.setEnabled(self.align_eyes.isChecked() and self.align_eyes.isEnabled())
        self.render_settings = self.settings
        if previous != self.settings and not self.restoring:
            self.invalidate_frame(clear=False)
            self.request_render()
        self.demo.mapping = mapping
        self.demo.calibration = self.settings.calibration
        self.output_size.setText(
            f"Выход: {width}×{height} · {'видеослот' if mapping == 'camera' else 'полный холст'}"
        )
        if persist and not self.restoring:
            self.save_timer.start(800)
        if not self.restoring:
            self.queue_obs_sync()

    def invalidate_frame(self, clear=False):
        self.session += 1
        self.publisher.reset(self.session, clear=clear)

    def arm_signal(self):
        self.update_tracking_quality({})
        if self.connection_wizard is not None and self.connection_wizard.active:
            self.connection_wizard.stop_probe()
        # A new source exits the old source's calibration view before opening its gate.
        if self.grid.isChecked():
            blocked = self.grid.blockSignals(True)
            self.grid.setChecked(False)
            self.grid.blockSignals(blocked)
        self.signal_monitor.arm()
        self.last_normal_output = QImage()
        self.watch_signal()

    def signal_policy_changed(self, *_):
        self.signal_policy = SignalPolicy(self.signal_timeout.value(), self.signal_fallback.currentData())
        self.watch_signal()
        if not self.restoring:
            self.save_timer.start(800)

    def signal_report(self):
        return {
            **self.signal_state,
            "emergency_black": self.emergency_black,
            "source_kind": self.active_source
            if self.active_source in ("camera", "screen", "photo") or self.video_path
            else "none",
            "policy": self.signal_policy.record(),
            "hold_reason": self.publisher.hold_reason,
            "tracking_evidence": self.last_tracking_quality,
            "composition_quality": self.last_composition,
            "scope": "Local source and processing continuity; does not verify OBS or Android",
        }

    def watch_signal(self):
        if self.closing:
            return
        expected = (
            False
            if self.active_source == "photo"
            else self.studio.screen.running and not self.studio.screen.paused
            if self.active_source == "screen"
            else self.camera_input.camera is not None and not self.camera_input.paused
            if self.active_source == "camera"
            else self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        suspended = self.connection_probe is not None or self.grid.isChecked()
        self.signal_state = self.signal_monitor.poll(expected, self.signal_policy.timeout, suspended)
        state = self.signal_state["state"]
        reason = (
            "emergency_black" if self.emergency_black else state if self.signal_state["blocked"] else None
        )
        action = (
            "black"
            if reason in ("emergency_black", "waiting", "source_stopped")
            else self.signal_policy.fallback
        )
        key = (reason, action, self.settings.width, self.settings.height) if reason else None
        if key is not None:
            if key != self.protection_key:
                import numpy as np

                image = self.last_normal_output if action == "freeze" else QImage()
                if (
                    action == "freeze"
                    and self.protection_key is not None
                    and self.protection_key[:2] == key[:2]
                ):
                    image = self.protection_image
                if image.isNull():
                    image = to_image(np.zeros((self.settings.height, self.settings.width, 3), np.uint8))
                elif image.width() != self.settings.width or image.height() != self.settings.height:
                    image = to_image(cv2.resize(to_array(image), (self.settings.width, self.settings.height)))
                self.protection_image = image
                self.publisher.hold(to_array(image), reason)
            self.display_output(self.protection_image)
        elif self.protection_key is not None:
            self.publisher.release()
            self.protection_image = QImage()
            if not self.last_normal_output.isNull():
                self.display_output(self.last_normal_output)
        self.protection_key = key
        labels = {
            "idle": "источник не выбран",
            "waiting": "ожидание свежего кадра",
            "live": "кадры поступают",
            "paused": "пауза / конец ролика",
            "calibrating": "проверочная сетка",
            "input_stalled": "входные кадры перестали поступать",
            "processing_stalled": "обработка не выдаёт свежие кадры",
            "source_error": "ошибка источника",
            "processing_error": "ошибка обработки",
            "source_stopped": "камера выключена",
        }
        text = "Выход закрыт вручную · чёрный кадр" if self.emergency_black else "Сигнал: " + labels[state]
        if self.active_source == "photo" and state == "paused" and not self.emergency_black:
            text = "Фото: изображение подготовлено"
        if reason and not self.emergency_black:
            text += " · " + ("чёрный кадр" if action == "black" else "удерживается последний кадр")
        self.signal_label.setText(text)
        self.signal_label.setStyleSheet("color: #fbbf24" if reason else "")
        self.black_button.setText("Вернуть видео" if self.emergency_black else "Чёрный кадр")
        self.editor.refresh_status()
        self.publisher.update_health(self.signal_report())

    def toggle_black_output(self):
        self.emergency_black = not self.emergency_black
        self.watch_signal()
        if self.emergency_black and self.connection_wizard is not None:
            self.connection_wizard.stop_probe()
        if (
            not self.emergency_black
            and self.publisher.hold_reason is None
            and not self.last_normal_output.isNull()
        ):
            self.publisher.publish(to_array(self.last_normal_output), self.session)

    def scan_interval(self):
        fast = (
            self.settings.motion_mode == "responsive"
            or self.settings.framing.enabled
            or any(slot.framing.enabled for slot in self.scene_slots)
        )
        return 120 if fast else 250

    def request_render(self):
        self.watch_signal()
        if self.connection_probe is not None:
            return
        if self.grid.isChecked():
            self.render_pending = False
            self.show_processed(calibration_card(self.settings.width, self.settings.height))
            if self.publisher.enabled.is_set():
                self.publisher.publish(to_array(self.last_output), self.session)
            return
        self.render_pending = True
        QTimer.singleShot(0, self.flush_render)

    def flush_render(self):
        if self.render_pending and not self.busy and not self.closing and self.latest_frame is not None:
            self.submit_frame(self.latest_frame, repaint=True)

    @staticmethod
    def time_label(milliseconds):
        seconds = max(0, int(milliseconds) // 1000)
        return (
            f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
            if seconds >= 3600
            else f"{seconds // 60:02}:{seconds % 60:02}"
        )

    def update_position(self, position, label_only=False):
        if self.active_source == "photo":
            self.position_label.setText("ФОТО")
            return
        if self.active_source != "file":
            self.position_label.setText("LIVE")
            return
        if not self.timeline.isSliderDown() and not label_only:
            self.timeline.setValue(position)
        self.position_label.setText(
            f"{self.time_label(position)} / {self.time_label(self.player.duration())}"
        )

    def duration_changed(self, duration):
        if self.active_source != "file":
            return
        self.timeline.setRange(0, max(0, duration))
        self.update_position(self.player.position())
        self.studio.duration_ready(duration)
        if duration > 0 and self.pending_scenario is not None:
            pending, self.pending_scenario = self.pending_scenario, None
            self.seek_to(pending["position"])
            selections = decode_selections(pending.get("selections", {}))
            self.click = selections if self.scene_slots else selections.get("single")

    def seek_from_keyboard(self):
        if not self.timeline.isSliderDown():
            self.seek_to(self.timeline.value())

    def seek_to(self, position):
        if self.active_source != "file" or self.player.source().isEmpty():
            return
        target = max(0, min(int(position), self.player.duration()))
        if target == self.player.position() and self.latest_frame is not None:
            return
        self.arm_signal()
        self.invalidate_frame(clear=True)
        self.latest_frame = None
        self.reset_pending = True
        self.object_seed = None
        self.object_seeds = {}
        self.render_pending = False
        self.seek_pending = True
        self.seek_target = target / 1000
        if self.object_sample_active and self.settings.subject_mode == "object":
            self.pending_object_sample = True
        self.status.setText("Перемотка…")
        self.player.setPosition(target)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.open_media(Path(urls[0].toLocalFile()))
            event.acceptProposedAction()

    def open_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать видео или фото",
            self.last_directory,
            "Видео и фото (*.mp4 *.mov *.m4v *.mkv *.avi *.webm *.mts *.m2ts *.ts *.mpeg *.mpg *.mxf *.wmv *.flv *.3gp *.ogv *.dv *.jpg *.jpeg *.png);;Фото (*.jpg *.jpeg *.png);;Все файлы (*)",
        )
        if file:
            self.open_media(Path(file))

    def open_media(self, path):
        if Path(path).suffix.lower() in PHOTO_EXTENSIONS:
            self.load_photo(Path(path))
        else:
            self.load_video(Path(path))

    def open_photo(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Выбрать фото", self.last_directory, "Фото (*.jpg *.jpeg *.png);;Все файлы (*)"
        )
        if file:
            self.load_photo(Path(file))

    def load_photo(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            self.status.setText("Фото не найдено: " + path.name)
            return
        self.photo_input.cancel()
        self.studio.screen.stop()
        self.camera_input.stop()
        self.file_epoch += 1
        self.active_source = "photo"
        self.player.stop()
        self.studio.new_source()
        self.photo_path = path
        self.photo_ready = False
        self.photo_opacity = None
        self.latest_frame = None
        self.latest_media_time = 0
        self.last_render_time = 0
        self.last_tracks = {}
        self.object_seed = None
        self.object_seeds = {}
        self.click = None
        self.reset_pending = True
        self.pending_scenario = None
        self.pending_object_sample = False
        self.object_sample_active = False
        self.seek_pending = False
        self.animation_timer.stop()
        self.last_directory = str(path.parent)
        self.filename.setText("Загрузка фото: " + path.name)
        self.timeline.setEnabled(False)
        self.timeline.setRange(0, 0)
        self.position_label.setText("ФОТО")
        self.telemetry.reset()
        self.metrics_started = False
        self.input_view.show_image(QImage())
        self.input_view.boxes = []
        self.input_view.focus = []
        self.input_view.polygons = []
        self.input_view.recovery = []
        self.input_view.markers = []
        self.recovery_candidates = []
        self.recovery_button.setEnabled(False)
        self.invalidate_frame(clear=True)
        self.arm_signal()
        self.status.setText("Читаю фото…")
        self.studio.photo_status.setText("Загрузка JPEG/PNG…")
        self.photo_input.load(path)
        self.editor.navigate(1)

    def photo_loaded(self, frame):
        if self.closing or self.active_source != "photo" or frame.path != self.photo_path:
            return
        self.latest_frame = frame.image
        self.photo_opacity = frame.opacity
        self.input_view.show_image(frame.preview)
        self.signal_monitor.received(0)
        self.telemetry.received += 1
        self.telemetry.accepted += 1
        self.filename.setText("Фото: " + frame.path.name)
        w, h = frame.original_size
        ih, iw = frame.image.shape[:2]
        self.studio.photo_status.setText(
            f"{w}×{h} → {iw}×{ih} · " + ("с прозрачностью" if frame.opacity is not None else "JPEG/PNG")
        )
        self.show_whole_photo()
        self.studio.restore_photo_state()
        self.render_pending = True
        self.flush_render()
        self.editor.open_section(1, "Фото")

    def show_whole_photo(self):
        if self.active_source != "photo" or self.latest_frame is None:
            return
        self.studio.selection_changed()
        self.subject_mode.setCurrentIndex(self.subject_mode.findData("object"))
        self.subject_region_combo.setCurrentIndex(0)
        self.studio.surface_lock.setChecked(False)
        self.studio.zoom.setValue(1)
        self.padding.setValue(0)
        self.click = {self.active_scene_slot: Rect(0, 0, 1, 1)} if self.scene_slots else Rect(0, 0, 1, 1)
        self.object_seed = (Rect(0, 0, 1, 1), 0)
        self.object_seeds[self.active_scene_slot or "single"] = self.object_seed
        self.reset_pending = True
        self.invalidate_frame(clear=False)
        self.request_render()

    def photo_face(self):
        if self.active_source != "photo" or self.latest_frame is None:
            self.status.setText("Сначала откройте фото")
            return
        self.subject_mode.setCurrentIndex(self.subject_mode.findData("face"))
        self.reset_tracking()
        self.editor.navigate(1)

    def photo_failed(self, message):
        if self.closing or self.active_source != "photo":
            return
        self.invalidate_frame(clear=False)
        self.render_pending = False
        self.signal_monitor.fail("source_error")
        self.watch_signal()
        self.status.setText(message)
        self.studio.photo_status.setText(message)

    def save_photo_image(self, path):
        if (
            self.active_source != "photo"
            or not self.photo_ready
            or self.last_output.isNull()
            or self.busy
            or self.render_pending
        ):
            raise ValueError("Сначала откройте и обработайте фото")
        destination = Path(path).resolve()
        if destination == self.photo_path:
            raise ValueError("Исходное фото нельзя перезаписывать")
        extension = destination.suffix.lower()
        if extension not in PHOTO_EXTENSIONS:
            raise ValueError("Выберите PNG или JPEG для результата")
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".framepilot-photo-", suffix=extension, dir=destination.parent)
        os.close(handle)
        temporary = Path(name)
        try:
            if not self.last_output.save(str(temporary), "PNG" if extension == ".png" else "JPEG", 95):
                raise ValueError("Не удалось сохранить изображение")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self.studio.photo_status.setText("Сохранено: " + destination.name)

    def save_photo_dialog(self):
        file, selected = QFileDialog.getSaveFileName(
            self,
            "Сохранить обработанное фото",
            str(
                (self.photo_path or Path(self.last_directory) / "photo.png").with_name(
                    (self.photo_path.stem if self.photo_path else "photo") + "-framed.png"
                )
            ),
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if file:
            path = Path(file)
            if not path.suffix:
                path = path.with_suffix(".jpg" if selected.startswith("JPEG") else ".png")
            try:
                self.save_photo_image(path)
            except (ValueError, OSError) as error:
                self.studio.photo_status.setText(str(error))

    def load_sample(self):
        source = Path(__file__).resolve().parent / "assets/astronaut.png"
        portrait = cv2.imread(str(source)) if source.exists() else None
        if portrait is None:
            self.status.setText("Тестовый портрет не найден; выберите свой видеофайл.")
            return
        if self.sample_directory is None:
            self.sample_directory = tempfile.TemporaryDirectory(prefix="framepilot-")
        path = Path(self.sample_directory.name) / "NASA-face-test.avi"
        if not path.exists():
            import numpy as np

            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (512, 512))
            if not writer.isOpened():
                self.status.setText("Не удалось создать тестовое видео; выберите свой файл.")
                return
            try:
                for index in range(144):
                    dx = round(30 * np.sin(index / 20))
                    shifted = cv2.warpAffine(portrait, np.float32([[1, 0, dx], [0, 1, 0]]), (512, 512))
                    writer.write(shifted)
            finally:
                writer.release()
        self.load_video(path)

    def load_video(self, file: Path):
        if not file.is_file():
            self.status.setText(f"Видеофайл не найден: {file.name}")
            return
        self.photo_input.cancel()
        self.photo_path = None
        self.photo_ready = False
        self.photo_opacity = None
        self.studio.screen.stop()
        self.studio.new_source()
        self.arm_signal()
        self.invalidate_frame(clear=True)
        self.pending_scenario = None
        self.pending_object_sample = False
        self.object_sample_active = False
        self.animation_timer.stop()
        self.part_button.setChecked(False)
        self.camera_input.stop()
        self.active_source = "file"
        self.timeline.setEnabled(True)
        self.video_path = file.resolve()
        self.grid.blockSignals(True)
        self.grid.setChecked(False)
        self.grid.blockSignals(False)
        self.object_seed = None
        self.object_seeds = {}
        self.last_output = QImage()
        self.last_tracks = {}
        self.last_render_time = 0.0
        self.recovery_candidates = []
        self.input_view.recovery = []
        self.recovery_button.setEnabled(False)
        self.file_epoch += 1
        self.studio.media.start(self.video_path)
        old_sink = self.sink
        old_sink.videoFrameChanged.disconnect()
        self.player.setVideoSink(None)
        self.player.stop()
        self.latest_frame = None
        self.reset_pending = True
        self.click = None
        self.last_directory = str(file.resolve().parent)
        self.render_pending = False
        self.latest_media_time = 0.0
        self.seek_pending = True
        self.seek_target = 0.0
        self.filename.setText(file.name)
        self.metric_start, self.processed = time.monotonic(), 0
        self.metrics_started = False
        self.telemetry.reset()
        for canvas in (self.input_view, self.output_view, self.output_window):
            canvas.show_image(QImage())
            canvas.boxes = []
            canvas.focus = []
            canvas.recovery = []
            canvas.markers = []
            canvas.polygons = []
        self.demo.image = QImage()
        self.player.setSource(QUrl.fromLocalFile(str(file.resolve())))
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(
            lambda frame, epoch=self.file_epoch: self.file_video_frame(frame, epoch)
        )
        self.player.setVideoSink(self.sink)
        old_sink.deleteLater()
        self.player.play()
        self.status.setText("Загрузка видео…")
        self.editor.navigate(1)

    def load_object_sample(self):
        from .samples import object_frame

        if self.sample_directory is None:
            self.sample_directory = tempfile.TemporaryDirectory(prefix="framepilot-")
        path = Path(self.sample_directory.name) / "synthetic-apple-test.avi"
        if not path.exists():
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
            if not writer.isOpened():
                self.status.setText("Не удалось создать тест предмета; выберите свой видеофайл")
                return
            try:
                for index in range(180):
                    writer.write(object_frame(index)[0])
            finally:
                writer.release()
        self.load_video(path)
        self.subject_mode.setCurrentIndex(self.subject_mode.findData("object"))
        self.subject_region_combo.setCurrentIndex(1)
        self.auto_framing_enabled.setChecked(False)
        self.motion_mode.setCurrentIndex(self.motion_mode.findData("responsive"))
        self.padding.setValue(0)
        self.object_sample_active = True
        self.pending_object_sample = True
        self.status.setText("Синтетический тест: отслеживается целое яблоко, выводится левая половина")

    def toggle_play(self):
        if self.active_source == "photo":
            return
        if self.active_source == "screen":
            self.studio.screen.toggle_pause()
            self.watch_signal()
            return
        if self.active_source == "camera":
            if self.camera_input.camera is not None:
                self.camera_input.toggle_pause()
            else:
                self.start_camera()
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.status.setText("Пауза")
        elif not self.player.source().isEmpty():
            self.player.play()

    def restart(self):
        if self.active_source == "photo":
            self.show_whole_photo()
            return
        if self.active_source != "file":
            self.reset_tracking()
            return
        if self.player.source().isEmpty():
            return
        self.seek_to(0)
        self.player.play()

    def reset_tracking(self):
        self.studio.selection_changed()
        self.object_sample_active = False
        self.object_seed = None
        self.invalidate_frame(clear=True)
        self.reset_pending = not bool(self.scene_slots)
        self.click = {"_reset": self.active_scene_slot} if self.scene_slots else None
        self.status.setText("Повторный поиск цели…")
        self.request_render()

    def select_face(self, point):
        if self.settings.subject_mode == "object":
            candidates = [rect for rect in self.recovery_candidates if rect.contains(point)]
            if candidates:
                self.select_object(candidates[0])
                return
            self.status.setText("Для выбора предмета выделите его прямоугольником")
            return
        self.studio.selection_changed()
        self.click = point
        self.invalidate_frame(clear=False)
        self.request_render()

    def select_object(self, rectangle):
        if self.latest_frame is None:
            return
        self.studio.selection_changed()
        self.object_sample_active = False
        self.pause_source()
        self.subject_mode.setCurrentIndex(1)
        self.object_seed = (rectangle, self.latest_media_time)
        self.object_seeds[self.active_scene_slot or "single"] = self.object_seed
        self.click = rectangle
        self.reset_pending = not bool(self.scene_slots)
        self.invalidate_frame(clear=False)
        self.request_render()

    def select_input_region(self, rectangle):
        if not self.part_button.isChecked():
            self.select_object(rectangle)
            return
        subject = self.input_view.boxes[0] if self.input_view.boxes else None
        if subject is None:
            self.status.setText("Сначала выделите целый предмет или выберите лицо/человека")
            return
        x, y = max(subject.x, rectangle.x), max(subject.y, rectangle.y)
        right, bottom = (
            min(subject.x + subject.w, rectangle.x + rectangle.w),
            min(subject.y + subject.h, rectangle.y + rectangle.h),
        )
        if right - x < subject.w * 0.02 or bottom - y < subject.h * 0.02:
            self.status.setText("Выделите часть внутри зелёной рамки цели")
            return
        selected_region = Rect(
            (x - subject.x) / subject.w,
            (y - subject.y) / subject.h,
            (right - x) / subject.w,
            (bottom - y) / subject.h,
        )
        if self.settings.surface_lock and self.settings.subject_mode == "object":
            quad = self.last_tracks.get(self.active_scene_slot or "single", {}).get("quad")
            if not valid_quad(quad):
                self.status.setText("Нет геометрии поверхности: подтвердите предмет или повторите анализ")
                return
            selected_region = inverse_region(quad, Rect(x, y, right - x, bottom - y))
            if min(selected_region.w, selected_region.h) < 0.02:
                self.status.setText("Выделите участок внутри поверхности предмета")
                return
        self.settings = replace(
            self.settings,
            subject_region=selected_region,
        )
        self.part_button.setChecked(False)
        self.sync_region_controls()
        self.read_settings()

    def begin_select_part(self, enabled):
        if enabled:
            self.pause_source()
            self.status.setText("Выделите нужную часть внутри зелёной рамки на входном изображении")

    def sync_region_controls(self):
        previous = self.subject_region_combo.blockSignals(True)
        try:
            index = next(
                (
                    index
                    for index in range(self.subject_region_combo.count())
                    if self.subject_region_combo.itemData(index) == self.settings.subject_region
                ),
                self.subject_region_combo.count() - 1,
            )
            self.subject_region_combo.setCurrentIndex(index)
        finally:
            self.subject_region_combo.blockSignals(previous)

    def subject_region_changed(self):
        region = self.subject_region_combo.currentData()
        if region is None:
            self.part_button.setChecked(True)
            return
        self.part_button.setChecked(False)
        self.settings = replace(self.settings, subject_region=region)
        self.read_settings()

    def framing_controls(self):
        return [self.whole_when, self.frame_threshold, self.framing_hysteresis, self.framing_transition]

    def load_framing_controls(self, rule):
        self.auto_framing_enabled.setChecked(rule.enabled)
        self.whole_when.setCurrentIndex(self.whole_when.findData(rule.whole_when))
        self.frame_threshold.setValue(rule.threshold * 100)
        self.framing_hysteresis.setValue(rule.hysteresis * 100)
        self.framing_transition.setValue(rule.transition)

    def remember_framing_threshold(self):
        self.frame_threshold.setValue(max(1, min(95, self.settings.target.area * 100)))

    def file_video_frame(self, frame, epoch):
        if epoch == self.file_epoch:
            self.video_frame(frame, source="file")

    def video_frame(self, frame, source="file"):
        if (
            self.closing
            or self.closing_requested
            or self.connection_probe is not None
            or source != self.active_source
            or self.grid.isChecked()
            or not frame.isValid()
        ):
            return
        if source == "camera" and self.camera_input.paused:
            return
        if (
            source == "file"
            and self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
            and not self.seek_pending
        ):
            return
        media_time = (
            time.monotonic() - self.camera_epoch
            if source == "camera"
            else (frame.startTime() / 1_000_000 if frame.startTime() >= 0 else self.player.position() / 1000)
        )
        if self.seek_pending and abs(media_time - self.seek_target) > 0.5:
            return
        if not self.input_ready(media_time):
            return
        image = frame.toImage()
        if image.isNull():
            return
        self.accept_input(to_array(image), media_time, source, accepted=True)

    def input_ready(self, media_time):
        if self.closing_requested:
            return False
        now = time.monotonic()
        self.signal_monitor.received(media_time, now)
        self.telemetry.received += 1
        target_fps = [30, 24, 15][self.fps_combo.currentIndex()]
        if self.busy:
            self.telemetry.busy_skips += 1
            return False
        if now - self.last_frame < 1 / target_fps - 0.002:
            self.telemetry.throttled += 1
            return False
        return True

    def accept_input(self, bgr, media_time, source, accepted=False):
        if (
            self.closing
            or self.closing_requested
            or source != self.active_source
            or self.connection_probe is not None
            or self.grid.isChecked()
        ):
            return
        if not accepted and not self.input_ready(media_time):
            return
        h, w = bgr.shape[:2]
        if max(w, h) > 960:
            scale = 960 / max(w, h)
            bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
        self.latest_frame = bgr
        self.latest_media_time = media_time
        if source == "file":
            self.studio.media.decoded()
            self.studio.recovery_frame_ready()
        if self.pending_object_sample and source == "file":
            from .samples import object_frame

            self.click = object_frame(round(media_time * 30))[1]
            self.object_seed = (self.click, media_time)
            self.object_seeds[self.active_scene_slot or "single"] = self.object_seed
            self.pending_object_sample = False
        self.seek_pending = False
        self.input_view.show_image(to_image(bgr))
        self.last_frame = time.monotonic()
        self.telemetry.accepted += 1
        self.submit_frame(bgr)

    def submit_frame(self, bgr, repaint=False):
        if self.busy or self.closing:
            return
        self.busy = True
        self.render_pending = False
        if self.active_source == "photo" and not self.photo_ready:
            repaint = False
        self.jobs.video.emit(
            (
                self.session,
                bgr,
                self.settings,
                self.click,
                self.reset_pending,
                repaint,
                self.latest_media_time,
                time.monotonic(),
                self.signal_policy.timeout,
                self.photo_opacity if self.active_source == "photo" else None,
                self.active_source == "photo",
            )
        )
        self.click = None
        self.reset_pending = False

    def frame_ready(self, result):
        self.busy = False
        session, output, face, faces, elapsed, backend, *features = result
        if session != self.session or self.closing:
            self.flush_render()
            return
        self.input_view.boxes = [face] if face else []
        self.input_view.candidates = faces
        eyes = features[0] if features else None
        metadata = features[1] if len(features) > 1 else {}
        completed = self.signal_monitor.complete(
            metadata.get("time", self.latest_media_time),
            metadata.get("submitted_at", time.monotonic()),
            self.signal_policy.timeout,
            repaint=metadata.get("repaint", False),
        )
        if self.active_source == "photo" and not self.photo_ready:
            self.photo_ready = completed
            if not completed:
                self.render_pending = True
        was_held = self.publisher.hold_reason is not None
        self.watch_signal()
        self.telemetry.complete(elapsed, face is not None, repaint=metadata.get("repaint", False))
        self.last_framing = metadata.get("framing", {})
        effective_region = Rect(
            *self.last_framing.get(
                "region",
                [
                    self.settings.subject_region.x,
                    self.settings.subject_region.y,
                    self.settings.subject_region.w,
                    self.settings.subject_region.h,
                ],
            )
        )
        self.input_view.focus = (
            [face.region(effective_region)] if face is not None and effective_region != WHOLE else []
        )
        if self.settings.framing.enabled:
            label = "целиком" if self.last_framing.get("whole") else "выбранная часть"
            suffix = " · переход…" if self.last_framing.get("transitioning") else ""
            self.framing_status.setText(
                f"Авто: {label}{suffix} · рамка {self.settings.target.area * 100:.1f}% окна"
            )
        self.last_tracks = metadata.get("tracks", {})
        self.studio.update_composition(metadata.get("composition", {}))
        surface = metadata.get("surface")
        self.input_view.polygons = [surface] if surface else []
        if surface:
            self.input_view.focus = []
        self.update_tracking_quality(metadata.get("tracking", {}))
        self.last_render_time = metadata.get("time", self.latest_media_time)
        self.recovery_candidates = metadata.get("recovery", [])
        self.recovery_button.setEnabled(len(self.recovery_candidates) == 1)
        self.recovery_button.setText(
            "Выберите нужного кандидата кликом на входном видео"
            if len(self.recovery_candidates) > 1
            else "Подтвердить найденную цель"
        )
        self.input_view.recovery = self.recovery_candidates
        self.input_view.markers = [eyes] if eyes is not None else []
        self.input_view.update()
        self.show_processed(output)
        if was_held and self.publisher.hold_reason is None:
            self.publisher.publish(output, session)
        self.processed += 1
        now = time.monotonic()
        if not self.metrics_started:
            self.metric_start, self.processed = now, 0
            self.metrics_started = True
            self.metrics.setText(f"Прогрев · {backend} · обработка {elapsed:.1f} мс")
        if now - self.metric_start >= 1:
            fps = self.processed / (now - self.metric_start)
            self.metrics.setText(
                f"{fps:.1f} кадров/с · обработка {elapsed:.1f} мс · {backend} · очередь ≤ 1 кадра"
            )
            self.metric_start, self.processed = now, 0
        if self.settings.subject_mode == "object":
            message = (
                "Предмет отслеживается" if face else "Предмет потерян или не выбран — выделите его мышкой"
            )
        elif self.settings.subject_mode in ("person", "upper"):
            message = "Человек отслеживается" if face else "Человек не подтверждён — выберите кликом"
        else:
            message = (
                "Лицо отслеживается" if face else "Лицо не подтверждено. Кликните по нему или сбросьте поиск."
            )
            if face and self.settings.eye_level is not None and eyes is None:
                message += " · глаза не найдены, центрирование по лицу"
        if self.recovery_candidates and face is None:
            message += " · найден похожий кандидат: проверьте и подтвердите"
        if self.active_source == "photo":
            message = "Фото · " + message
            self.metrics.setText(f"Фото · обработка {elapsed:.1f} мс · {backend}")
        if (
            (
                self.active_source == "file"
                and self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
            )
            or self.camera_input.paused
            or (self.active_source == "screen" and self.studio.screen.paused)
        ):
            message = "Пауза · " + message
        self.status.setText(message)
        paused = (
            self.active_source == "photo"
            or (
                self.active_source == "file"
                and self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
            )
            or self.camera_input.paused
            or (self.active_source == "screen" and self.studio.screen.paused)
        )
        if (
            metadata.get("animating")
            and paused
            and not self.animation_timer.isActive()
            and not self.closing_requested
        ):
            self.animation_timer.start(round(1000 / [30, 24, 15][self.fps_combo.currentIndex()]))
        self.flush_render()

    def video_error(self, session, message):
        self.busy = False
        if self.closing_requested:
            self.render_pending = False
        if session != self.session:
            self.flush_render()
            return
        if session == self.session and not self.closing:
            self.pause_source()
            self.signal_monitor.fail("processing_error")
            self.watch_signal()
            self.status.setText(f"Ошибка обработки: {message}")

    def player_error(self, error, message):
        if self.active_source != "file" or self.closing_requested:
            return
        if error != QMediaPlayer.Error.NoError:
            if self.active_source == "file":
                self.studio.media.preview_error = message
                self.studio.media.refresh()
                self.signal_monitor.fail("source_error")
                self.watch_signal()
            self.status.setText(f"Не удалось воспроизвести видео: {message}")

    def media_status(self, status):
        if self.active_source != "file" or self.closing_requested:
            return
        if (
            status == QMediaPlayer.MediaStatus.EndOfMedia
            and self.player.mediaStatus() == status
            and self.player.playbackState() != QMediaPlayer.PlaybackState.PausedState
            and self.loop.isChecked()
        ):
            self.restart()

    def show_output(self):
        self.output_window.show()
        self.output_window.raise_()

    def start_local_output(self, copy_address=False):
        if self.closing:
            return
        try:
            url = self.publisher.start(self.preferred_port)
            self.preferred_port = self.publisher.server.server_port
            if copy_address:
                QApplication.clipboard().setText(url)
            self.local_url.setText(url)
            self.local_url.setToolTip(
                "Адрес скопирован" if copy_address else "Добавьте этот адрес в источник «Браузер» в OBS"
            )
            if not self.last_output.isNull():
                self.publisher.publish(to_array(self.last_output))
            self.save_preferences()
            self.queue_obs_sync()
        except OSError as error:
            self.local_url.setText(f"Не удалось открыть локальный выход: {error}")

    def stop_local_output(self):
        self.publisher.stop()
        self.local_url.clear()
        self.obs_pending = None
        self.save_preferences()
        if self.obs_connected and not self.obs_busy:
            self.obs_busy = True
            self.jobs.obs.emit(("disconnect", ()))

    def open_output_url(self):
        if self.publisher.url:
            QDesktopServices.openUrl(QUrl(self.publisher.url))

    def configure_obs(self):
        dialog = OBSSetupDialog(self)
        if dialog.exec() != OBSSetupDialog.DialogCode.Accepted:
            return
        self.connect_obs(dialog.port.value(), dialog.password.text(), dialog.activate.isChecked())
        dialog.password.clear()

    def configure_camera_name(self):
        if sys.platform != "win32":
            self.status.setText("Изменение имени виртуальной камеры доступно в Windows.")
            return
        existing = getattr(self, "camera_name_dialog", None)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return
        from .camera_name_dialog import CameraNameDialog
        from .windows_camera_names import CameraNameError

        try:
            dialog = CameraNameDialog(self.preferences.path.with_name("camera-name-original.json"), self)
        except (CameraNameError, OSError) as error:
            self.status.setText(str(error))
            return
        self.camera_name_dialog = dialog
        dialog.exec()
        self.camera_name_dialog = None
        dialog.deleteLater()
        self.refresh_cameras()
        if self.connection_wizard is not None:
            self.connection_wizard.refresh_cameras()

    def connect_obs(self, port, password, activate=False):
        if self.obs_busy:
            return
        self.start_local_output()
        if not self.publisher.url:
            return
        self.obs_button.setEnabled(False)
        self.obs_busy = True
        self.obs_status.setText("Настройка сцены в OBS…")
        self.jobs.obs.emit(
            (
                "configure",
                (
                    port,
                    password,
                    self.publisher.url,
                    self.settings.width,
                    self.settings.height,
                    activate,
                    [30, 24, 15][self.fps_combo.currentIndex()],
                ),
            )
        )

    def open_connection_wizard(self):
        from .connections import ConnectionWizard

        self.editor.navigate(3)
        wizard = ConnectionWizard(self)
        wizard.exec()
        wizard.deleteLater()

    def obs_finished(self, state):
        self.obs_button.setEnabled(True)
        self.obs_busy = False
        self.obs_connected = state["connected"]
        if "synced" in state:
            self.obs_last_synced = tuple(state["synced"])
        if self.obs_connected and self.publisher.url is None:
            self.obs_busy = True
            self.jobs.obs.emit(("disconnect", ()))
            return
        self.obs_status.setText(
            "OBS подключена · "
            + ("источник виден" if state.get("source_showing") else "выберите сцену FramePilot")
            + " · камера "
            + ("включена" if state.get("virtual_camera") else "выключена")
            if self.obs_connected
            else "OBS отключена"
        )
        self.flush_obs_sync()
        self.queue_obs_sync()

    def obs_failed(self, message):
        self.obs_busy = False
        self.obs_connected = False
        self.obs_pending = None
        self.obs_last_synced = None
        self.obs_button.setEnabled(True)
        self.obs_status.setText(message)

    def queue_obs_sync(self):
        if self.obs_connected and self.publisher.url:
            desired = (
                self.publisher.url,
                self.settings.width,
                self.settings.height,
                [30, 24, 15][self.fps_combo.currentIndex()],
            )
            if desired == self.obs_last_synced:
                self.obs_pending = None
                self.obs_sync_timer.stop()
            elif desired != self.obs_pending:
                self.obs_pending = desired
                self.obs_sync_timer.start(400)

    def flush_obs_sync(self):
        if self.obs_connected and not self.obs_busy and self.obs_pending is not None:
            pending, self.obs_pending = self.obs_pending, None
            if pending == self.obs_last_synced:
                return
            self.obs_busy = True
            self.jobs.obs.emit(("sync", pending))

    def poll_obs(self):
        if self.obs_connected and not self.obs_busy:
            self.obs_busy = True
            self.jobs.obs.emit(("status", ()))

    def show_demo(self):
        self.demo.show()
        self.demo.raise_()
        self.window_combo.setCurrentIndex(0)

    def save_debug_case(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Сохранить кадры и параметры ошибки", self.last_directory
        )
        if directory:
            from .debug_bundle import save_debug_bundle

            try:
                folder = save_debug_bundle(self, directory)
                self.status.setText("Пример ошибки сохранён: " + folder.name)
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
            except (OSError, TypeError, ValueError) as error:
                self.status.setText("Не удалось сохранить пример: " + str(error))

    def refresh_windows(self):
        if self.listing:
            return
        if sys.platform not in ("darwin", "win32"):
            self.window_status.setText("Захват внешних окон поддерживается в macOS и Windows.")
            return
        if sys.platform == "darwin":
            import Quartz

            if not Quartz.CGPreflightScreenCaptureAccess():
                if not Quartz.CGRequestScreenCaptureAccess():
                    self.window_status.setText(
                        "Нет доступа к внешним окнам. Нажмите «Разрешение записи экрана…», "
                        "включите FramePilot и перезапустите его, затем обновите список окон."
                    )
                    return
        self.listing = True
        self.refresh_button.setEnabled(False)
        self.window_status.setText("Получение списка окон…")
        self.jobs.list_windows.emit()

    def open_screen_permission_settings(self):
        QDesktopServices.openUrl(
            QUrl(
                "ms-settings:privacy"
                if sys.platform == "win32"
                else "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
            )
        )

    def windows_ready(self, windows):
        old = self.window_combo.currentData()
        old_id = old.identifier if old else None
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        self.window_combo.addItem("Тестовое приложение", None)
        chosen = 0
        for index, window in enumerate(windows, start=1):
            self.window_combo.addItem(window.title, window)
            if window.identifier == old_id:
                chosen = index
        self.window_combo.setCurrentIndex(chosen)
        self.window_combo.blockSignals(False)
        self.listing = False
        self.refresh_button.setEnabled(True)
        current = self.window_combo.currentData()
        if old_id != (current.identifier if current else None):
            self.target_changed()
        if current is None:
            self.window_status.setText(
                f"Доступно внешних окон: {len(windows)}. Выберите нужное в списке выше. "
                "Сейчас выбрано тестовое приложение."
            )

    def target_changed(self, *_):
        self.generation += 1
        self.stable = StableTarget()
        self.slots = []
        self.current_slot = 0
        self.manual_slot = False
        self.roi = None
        self.target_view.show_image(QImage())
        self.target_view.boxes = []
        self.target_view.candidates = []
        self.window_status.setText("Ожидание устойчивой рамки…")
        key, _ = self.current_window_profile_key()
        self.apply_named_profile(key)
        QTimer.singleShot(0, self.scan_window)

    def scan_window(self):
        if self.closing or self.closing_requested or self.scan_busy or self.listing:
            return
        window = self.window_combo.currentData()
        internal = None
        if window is None:
            if not self.demo.isVisible():
                return
            internal = to_array(self.demo.grab().toImage())
        self.scan_busy = True
        self.jobs.scan.emit((self.generation, window, internal, self.roi, self.sensitivity.currentData()))

    def scan_ready(self, result):
        self.scan_busy = False
        generation, image, slots = result
        if generation != self.generation or self.closing or self.closing_requested:
            return
        self.slot_image_size = image.shape[1], image.shape[0]
        self.target_view.show_image(to_image(image))
        self.slots = slots
        self.target_view.candidates = [slot.rect for slot in slots]
        if self.manual_slot or not self.auto_slot.isChecked():
            return
        if self.scene_slots:
            self.update_scene_positions(slots)
            return
        accepted = self.stable.update([slot.rect for slot in slots])
        if accepted is not None:
            slot = min(slots, key=lambda s: abs(s.rect.x - accepted.x) + abs(s.rect.y - accepted.y))
            self.settings = replace(self.settings, target=accepted, shape=slot.shape)
            self.target_view.boxes = [accepted]
            self.read_settings(persist=False)
            self.window_status.setText(f"Рамка подтверждена · {slot.shape} · кандидатов: {len(slots)}")
        elif not slots:
            self.window_status.setText(
                "Рамка не найдена. Сохраняется последняя; можно выделить другую мышкой."
            )
        else:
            self.window_status.setText("Проверка рамки на следующем снимке…")

    def capture_error(self, generation, message):
        if generation == -1:
            self.listing = False
            self.refresh_button.setEnabled(True)
        else:
            self.scan_busy = False
        if generation in (-1, self.generation):
            self.window_status.setText(f"Захват окна: {message}")

    def manual_frame(self, rect):
        if self.roi_button.isChecked():
            self.roi = rect
            self.roi_button.setChecked(False)
            self.detection_changed()
            self.window_status.setText("Поиск ограничен выделенной областью. «Всё окно» снимает ограничение.")
            return
        # Invalidate an outstanding automatic scan before accepting a manual override.
        self.generation += 1
        self.manual_slot = True
        self.settings = replace(self.settings, target=rect, shape="rectangle")
        self.target_view.boxes = [rect]
        self.target_view.update()
        self.read_settings()
        if self.scene_slots:
            self.target_view.boxes = [slot.target for slot in self.scene_slots]
            self.target_view.update()
        self.window_status.setText("Ручная рамка. Нажмите «Автопоиск», чтобы снова следить автоматически.")

    def choose_slot(self, point):
        choices = [(i, slot) for i, slot in enumerate(self.slots) if slot.rect.contains(point)]
        if not choices or self.roi_button.isChecked():
            return
        self.current_slot, slot = min(choices, key=lambda item: item[1].rect.area)
        self.generation += 1
        self.stable = StableTarget()
        self.stable.accepted = self.stable.pending = slot.rect
        self.settings = replace(self.settings, target=slot.rect, shape=slot.shape)
        self.target_view.boxes = [slot.rect]
        self.manual_slot = False
        self.read_settings()
        self.target_view.update()

    def toggle_roi_selection(self):
        if self.roi_button.isChecked():
            self.window_status.setText("Выделите мышкой область, внутри которой нужно искать рамку.")

    def clear_roi(self):
        self.roi = None
        self.roi_button.setChecked(False)
        self.detection_changed()

    def detection_changed(self, *_):
        if self.restoring:
            return
        self.generation += 1
        self.stable = StableTarget()
        self.manual_slot = False
        self.save_timer.start(800)

    def resume_auto(self):
        self.manual_slot = False
        self.stable = StableTarget()
        self.auto_slot.setChecked(True)

    def next_slot(self):
        if not self.slots:
            return
        self.generation += 1
        self.current_slot = (self.current_slot + 1) % len(self.slots)
        slot = self.slots[self.current_slot]
        self.stable = StableTarget()
        self.stable.accepted = slot.rect
        self.stable.pending = slot.rect
        self.settings = replace(self.settings, target=slot.rect, shape=slot.shape)
        self.target_view.boxes = [slot.rect]
        self.manual_slot = False
        self.read_settings()

    def refresh_cameras(self):
        old = self.camera_combo.currentData()
        old_id = bytes(old.id()) if old is not None else None
        entries = registered_names()
        devices = self.camera_input.devices()
        remember_obs_devices(devices, entries)
        self.camera_combo.clear()
        for device in devices:
            self.camera_combo.addItem(camera_label(device, entries), device)
            if old_id is not None and bytes(device.id()) == old_id:
                self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("Камеры не найдены", None)

    def start_camera(self, device_id=None):
        if isinstance(device_id, bool):
            device_id = None
        if self.camera_combo.currentData() is None:
            self.refresh_cameras()
        if isinstance(device_id, str) and device_id:
            for index in range(self.camera_combo.count()):
                device = self.camera_combo.itemData(index)
                if device is not None and bytes(device.id()).hex() == device_id:
                    self.camera_combo.setCurrentIndex(index)
                    break
            else:
                self.status.setText("Камера сценария недоступна; выберите устройство вручную")
                return
        device = self.camera_combo.currentData()
        if device is None:
            self.status.setText("Камера не найдена")
            return
        self.photo_input.cancel()
        self.photo_path = None
        self.photo_ready = False
        self.photo_opacity = None
        self.studio.screen.stop()
        self.studio.new_source()
        self.arm_signal()
        self.player.pause()
        self.active_source = "camera"
        self.camera_epoch = time.monotonic()
        self.telemetry.reset()
        self.latest_frame = None
        self.reset_pending = True
        self.seek_pending = False
        self.pending_scenario = None
        self.pending_object_sample = False
        self.object_sample_active = False
        self.animation_timer.stop()
        self.object_seed = None
        self.object_seeds = {}
        self.click = None
        self.last_tracks = {}
        self.last_render_time = 0.0
        self.recovery_candidates = []
        self.input_view.recovery = []
        self.recovery_button.setEnabled(False)
        self.invalidate_frame(clear=True)
        self.timeline.setEnabled(False)
        self.position_label.setText("LIVE")
        self.filename.setText("Камера: " + device.description())
        self.editor.navigate(1)
        self.camera_input.start(device)

    def stop_camera(self):
        self.camera_input.stop()
        if self.active_source == "camera":
            self.signal_monitor.fail("source_stopped")
            self.watch_signal()
            self.active_source = "file"
            self.latest_frame = None
            self.reset_pending = True
            self.timeline.setEnabled(True)
            self.invalidate_frame(clear=False)
            self.filename.setText(self.video_path.name if self.video_path else "Видео не выбрано")
            self.status.setText("Камера выключена")

    def camera_message(self, text):
        if self.closing:
            return
        if self.active_source == "camera" and text.startswith(
            ("Ошибка камеры", "Доступ к камере", "Разрешите")
        ):
            self.signal_monitor.fail("source_error")
            self.watch_signal()
        if text.startswith("Ошибка камеры"):
            self.camera_input.stop()
            self.invalidate_frame(clear=False)
            self.render_pending = False
        self.status.setText(text)

    def pause_source(self):
        if self.active_source == "screen":
            if self.studio.screen.running and not self.studio.screen.paused:
                self.studio.screen.toggle_pause()
        elif self.active_source == "camera" and self.camera_input.camera is not None:
            if not self.camera_input.paused:
                self.camera_input.toggle_pause()
        else:
            self.player.pause()

    def confirm_recovery(self):
        if len(self.recovery_candidates) != 1:
            return
        candidate = self.recovery_candidates[0]
        self.pause_source()
        if self.settings.subject_mode == "object":
            self.select_object(candidate)
        else:
            self.select_face(candidate.center)

    def update_tracking_quality(self, evidence):
        self.last_tracking_quality = evidence
        self.quality_label.setVisible(bool(evidence))
        if not evidence:
            return
        state = evidence.get("state", "unselected")
        labels = {
            "static": "выбранная область фото",
            "selected": "цель выбрана вручную",
            "stable": "устойчивые ориентиры",
            "weak": "ориентиры ограничены — следите за рамкой",
            "recovering": "проверка восстановления по двум кадрам",
            "ambiguous": "несколько похожих предметов — выберите нужный кликом",
            "lost": "цель потеряна",
            "unselected": "выделите предмет на входном видео",
        }
        detail = [
            "Это оценка признаков трекера, не вероятность распознавания личности.",
            "Метод: " + evidence.get("basis", "—"),
        ]
        for key, label in (
            ("points", "Опорных точек"),
            ("inlier_ratio", "Доля согласованных точек"),
            ("template_error", "Ошибка образца"),
            ("appearance_distance", "Различие цветов"),
            ("rotation_degrees", "Поворот, °"),
        ):
            if key in evidence:
                detail.append(f"{label}: {evidence[key]}")
        self.quality_label.setText("Контроль цели: " + labels.get(state, state))
        self.quality_label.setToolTip("\n".join(detail))
        color = "#51e0b4" if state in ("stable", "selected", "static") else "#fbbf24"
        self.quality_label.setStyleSheet("color: " + color)

    def current_slot_config(self, identifier=None, target=None):
        return SceneSlot(
            identifier or uuid.uuid4().hex,
            target or self.settings.target,
            self.settings.shape,
            self.settings.subject_mode,
            self.settings.padding,
            self.settings.fill,
            self.settings.eye_level,
            self.settings.max_zoom,
            self.settings.subject_region,
            self.settings.framing,
            frame_mask=self.settings.frame_mask,
            framing_zoom=self.settings.framing_zoom,
            surface_lock=self.settings.surface_lock,
        )

    def refresh_scene_combo(self):
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        if not self.scene_slots:
            self.scene_combo.addItem("Одна рамка", "")
        names = {"face": "лицо", "object": "предмет", "upper": "по пояс", "person": "целиком"}
        for index, slot in enumerate(self.scene_slots, 1):
            self.scene_combo.addItem(f"Рамка {index} · {names[slot.subject_mode]}", slot.identifier)
        index = self.scene_combo.findData(self.active_scene_slot)
        self.scene_combo.setCurrentIndex(max(0, index))
        self.scene_combo.blockSignals(False)

    def load_slot_controls(self, slot):
        widgets = [
            self.subject_mode,
            self.padding,
            self.fit_combo,
            self.align_eyes,
            self.eye_level,
            self.max_zoom,
            self.mode,
            self.subject_region_combo,
            self.auto_framing_enabled,
            *self.framing_controls(),
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.settings = replace(
                self.settings, target=slot.target, shape=slot.shape, subject_region=slot.subject_region
            )
            self.settings = replace(
                self.settings,
                frame_mask=slot.frame_mask,
                framing_zoom=slot.framing_zoom,
                surface_lock=slot.surface_lock,
            )
            self.studio.sync_controls(self.settings)
            self.sync_region_controls()
            self.load_framing_controls(slot.framing)
            self.subject_mode.setCurrentIndex(self.subject_mode.findData(slot.subject_mode))
            self.padding.setValue(slot.padding)
            self.fit_combo.setCurrentIndex(int(slot.fill))
            self.align_eyes.setChecked(slot.eye_level is not None)
            if slot.eye_level is not None:
                self.eye_level.setValue(slot.eye_level)
            self.max_zoom.setValue(slot.max_zoom)
            self.mode.setCurrentIndex(1)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.click = None
        self.object_seed = self.object_seeds.get(self.active_scene_slot)
        self.read_settings()
        self.refresh_scene_combo()
        self.target_view.boxes = [item.target for item in self.scene_slots]
        self.target_view.update()
        self.studio.update_markers()

    def select_scene_slot(self, *_):
        identifier = self.scene_combo.currentData()
        slot = next((item for item in self.scene_slots if item.identifier == identifier), None)
        if slot is not None:
            self.active_scene_slot = identifier
            self.load_slot_controls(slot)

    def add_scene_slot(self):
        if len(self.scene_slots) >= 4:
            self.window_status.setText("Поддерживается до четырёх рамок")
            return
        if not self.scene_slots:
            self.scene_slots = [self.current_slot_config(target=Rect(0.05, 0.1, 0.4, 0.8))]
        target = (
            Rect(0.55, 0.1, 0.4, 0.8)
            if len(self.scene_slots) == 1
            else (Rect(0.05, 0.55, 0.4, 0.4) if len(self.scene_slots) == 2 else Rect(0.55, 0.55, 0.4, 0.4))
        )
        if len(self.scene_slots) == 2:
            self.scene_slots = [
                replace(item, target=Rect(0.05 + index * 0.5, 0.05, 0.4, 0.4))
                for index, item in enumerate(self.scene_slots)
            ]
        slot = self.current_slot_config(target=target)
        self.scene_slots.append(slot)
        self.active_scene_slot = slot.identifier
        self.auto_slot.setChecked(False)
        self.load_slot_controls(slot)
        self.window_status.setText(
            "Выберите цель для активной рамки; положение задаётся в предпросмотре окна назначения"
        )

    def remove_scene_slot(self):
        self.scene_slots = [slot for slot in self.scene_slots if slot.identifier != self.active_scene_slot]
        self.object_seeds.pop(self.active_scene_slot, None)
        self.slot_stabilizers = {}
        if self.scene_slots:
            self.active_scene_slot = self.scene_slots[0].identifier
            self.load_slot_controls(self.scene_slots[0])
        else:
            self.active_scene_slot = ""
            self.reset_pending = True
            self.read_settings()
            self.refresh_scene_combo()

    def scene_from_detected(self):
        if not self.slots:
            self.window_status.setText("Сначала откройте окно назначения и найдите рамки")
            return
        self.scene_slots = [
            replace(self.current_slot_config(target=slot.rect), shape=slot.shape) for slot in self.slots[:4]
        ]
        self.active_scene_slot = self.scene_slots[0].identifier
        self.slot_stabilizers = {}
        self.load_slot_controls(self.scene_slots[0])

    def update_scene_positions(self, candidates):
        remaining = list(candidates)
        updated = []
        from .geometry import iou

        for slot in self.scene_slots:
            match = max(remaining, key=lambda value: iou(value.rect, slot.target), default=None)
            if match is not None and iou(match.rect, slot.target) > 0.2:
                stable = self.slot_stabilizers.setdefault(slot.identifier, StableTarget())
                accepted = stable.update([match.rect])
                if accepted is not None:
                    slot = replace(slot, target=accepted, shape=match.shape)
                remaining.remove(match)
            updated.append(slot)
        self.scene_slots = updated
        active = next((slot for slot in updated if slot.identifier == self.active_scene_slot), updated[0])
        self.settings = replace(self.settings, target=active.target, shape=active.shape)
        self.read_settings(persist=False)
        self.target_view.boxes = [slot.target for slot in updated]
        self.target_view.update()
        self.window_status.setText(f"Композиция: {len(updated)} рамок; позиции без совпадения сохраняются")

    def choose_background_color(self):
        color = QColorDialog.getColor(QColor(self.settings.background_color), self, "Цвет фона композиции")
        if color.isValid():
            self.settings = replace(self.settings, background_color=color.name(), background_image="")
            self.read_settings()

    def choose_background_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Фон композиции", "", "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if file:
            self.settings = replace(self.settings, background_image=file)
            self.read_settings()

    def clear_background_image(self):
        self.settings = replace(self.settings, background_image="")
        self.read_settings()

    def current_selections(self):
        return {
            identifier: (
                QuadSelection(tuple(tuple(float(v) for v in p) for p in info["quad"]))
                if info["mode"] == "object"
                and info.get("composition", {}).get("surface_locked")
                and valid_quad(info.get("quad"))
                else info["subject"]
                if info["mode"] == "object"
                else info["subject"].center
            )
            for identifier, info in self.last_tracks.items()
            if info.get("subject") is not None
        }

    def install_shortcuts(self):
        shortcuts = {
            "Ctrl+E": self.start_export,
            "Ctrl+Shift+X": self.toggle_black_output,
            "Ctrl+Shift+G": lambda: self.grid.setChecked(not self.grid.isChecked()),
            "Ctrl+Shift+B": lambda: self.remove_background.setChecked(not self.remove_background.isChecked()),
            "Ctrl+Shift+K": lambda: (
                self.stop_camera() if self.active_source == "camera" else self.start_camera()
            ),
        }
        for index in range(4):
            shortcuts[f"Ctrl+{index + 1}"] = lambda i=index: self.scene_combo.setCurrentIndex(
                min(i, self.scene_combo.count() - 1)
            )
            shortcuts[f"Ctrl+Alt+{index + 1}"] = lambda i=index: self.apply_scenario_index(i)
        for sequence, callback in shortcuts.items():
            shortcut = QShortcut(QKeySequence(sequence), self, activated=callback)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)

    def show_shortcuts(self):
        QMessageBox.information(
            self,
            "Горячие клавиши FramePilot",
            "Cmd+O — открыть видео\nПробел — пуск/пауза\nCmd+E — экспорт MP4\n"
            "Cmd+1…4 — активная рамка\nCmd+Option+1…4 — сценарий\nCmd+Shift+G — проверочная сетка\n"
            "Cmd+Shift+B — вырезание людей\nCmd+Shift+K — включить/выключить камеру\n"
            "Cmd+Shift+X — чёрный кадр / вернуть видео в выход\n\n"
            "Комбинации действуют в окнах FramePilot, не являются глобальными для других приложений.",
        )

    def refresh_scenarios(self):
        self.scenario_combo.clear()
        self.scenario_combo.addItem("Выберите сценарий…", None)
        for key, value in self.scenarios.items():
            if isinstance(value, dict) and isinstance(value.get("name"), str):
                self.scenario_combo.addItem(value["name"], key)

    def save_scenario(self, name=None):
        source = self.photo_path if self.active_source == "photo" else self.video_path
        if self.active_source in ("file", "photo") and source is None:
            self.status.setText("Откройте видео или включите камеру перед сохранением сценария")
            return
        if not isinstance(name, str):
            name, ok = QInputDialog.getText(self, "Сохранить сценарий", "Название:")
            if not ok or not name.strip():
                return
        name = name.strip()[:120]
        key = next(
            (key for key, value in self.scenarios.items() if value.get("name") == name), uuid.uuid4().hex
        )
        window_key, _ = self.current_window_profile_key()
        device = self.camera_input.device
        screen_source = self.studio.screen.source if self.active_source == "screen" else None
        self.scenarios[key] = {
            "name": name,
            "profile": self.capture_profile(),
            "source": self.active_source,
            "path": str(source or ""),
            "position": max(0, round(self.last_render_time * 1000)),
            "camera_device": bytes(device.id()).hex() if device is not None else "",
            "screen_source": {"kind": screen_source.kind, "id": screen_source.identifier}
            if screen_source
            else None,
            "window": window_key,
            "selections": encode_selections(self.current_selections())
            if self.active_source in ("file", "photo")
            else {},
        }
        self.refresh_scenarios()
        self.scenario_combo.setCurrentIndex(self.scenario_combo.findData(key))
        self.save_preferences()

    def apply_scenario_index(self, index):
        if index < len(self.scenarios):
            self.scenario_combo.setCurrentIndex(index + 1)
            self.apply_scenario()

    def apply_scenario(self):
        entry = self.scenarios.get(self.scenario_combo.currentData())
        if not isinstance(entry, dict):
            return
        try:
            profile = validate_profile(entry["profile"])
            selections = decode_selections(entry.get("selections", {}))
            if (
                entry.get("source") not in ("file", "photo", "camera", "screen")
                or type(entry.get("position", 0)) is not int
                or entry.get("position", 0) < 0
            ):
                raise ValueError("Неверные параметры сценария")
            if entry["source"] in ("file", "photo") and not Path(entry.get("path", "")).is_file():
                raise ValueError("Медиафайл сценария не найден")
            for index in range(self.window_combo.count()):
                window = self.window_combo.itemData(index)
                key = window.profile_key if window is not None else "demo"
                if key == entry.get("window"):
                    self.window_combo.setCurrentIndex(index)
                    break
            if entry["source"] == "file":
                self.load_video(Path(entry["path"]))
                self.apply_profile(profile)
                self.pending_scenario = {
                    "position": entry.get("position", 0),
                    "selections": encode_selections(selections),
                }
                if self.player.duration() > 0:
                    self.duration_changed(self.player.duration())
            elif entry["source"] == "photo":
                self.load_photo(Path(entry["path"]))
                self.studio.pending_photo_state = {
                    "source": str(self.photo_path),
                    "profile": profile,
                    "selections": encode_selections(selections),
                }
            elif entry["source"] == "screen":
                source = entry.get("screen_source") or {}
                if not isinstance(source, dict):
                    raise ValueError("Неверный экранный источник сценария")
                for i in range(self.studio.source_combo.count()):
                    candidate = self.studio.source_combo.itemData(i)
                    if (
                        candidate
                        and candidate.kind == source.get("kind")
                        and candidate.identifier == source.get("id")
                    ):
                        self.studio.source_combo.setCurrentIndex(i)
                        break
                else:
                    raise ValueError(
                        "Обновите список экранов/окон и сохраните сценарий заново, если окно изменилось"
                    )
                self.apply_profile(profile)
                self.studio.start_screen()
            else:
                self.apply_profile(profile)
                self.start_camera(entry.get("camera_device", ""))
        except (ValueError, TypeError, KeyError) as error:
            self.status.setText("Сценарий не применён: " + str(error))

    def remove_scenario(self):
        key = self.scenario_combo.currentData()
        if key is not None:
            self.scenarios.pop(key, None)
            self.refresh_scenarios()
            self.save_preferences()

    def show_processed(self, output):
        self.last_normal_output = to_image(output)
        self.display_output(
            self.protection_image if not self.protection_image.isNull() else self.last_normal_output
        )

    def display_output(self, image):
        self.last_output = image
        self.output_window.show_image(self.last_output)
        self.update_output_preview()
        self.demo.image = self.last_output
        self.demo.update()
        if self.studio.recorder.active:
            self.studio.recorder.push(to_array(self.last_output))

    def update_output_preview(self, *_):
        if self.last_output.isNull():
            return
        image = self.last_output
        if self.receiver_preview.isChecked() and self.settings.mapping == "camera":
            image = to_image(received_preview(to_array(image), self.settings.calibration))
        self.output_view.show_image(image)

    def configure_calibration(self):
        dialog = CalibrationDialog(self.settings.calibration, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self.settings = replace(self.settings, calibration=dialog.value())
            self.read_settings()
            self.update_output_preview()

    def toggle_grid(self, checked):
        import numpy as np

        self.pause_source()
        self.invalidate_frame(clear=False)
        if not checked and self.latest_frame is None:
            self.show_processed(np.zeros((self.settings.height, self.settings.width, 3), np.uint8))
            self.publisher.publish(to_array(self.last_output), self.session)
            return
        self.request_render()

    def current_window_profile_key(self):
        window = self.window_combo.currentData()
        return (window.profile_key, window.title) if window else ("demo", "Тестовое приложение")

    def refresh_window_profiles(self):
        self.window_profile_combo.blockSignals(True)
        self.window_profile_combo.clear()
        self.window_profile_combo.addItem("Профиль окна…", None)
        for key, entry in self.window_profiles.items():
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                self.window_profile_combo.addItem(entry["name"], key)
        self.window_profile_combo.blockSignals(False)

    def remember_window_profile(self):
        key, name = self.current_window_profile_key()
        self.window_profiles[key] = {"name": name, "profile": self.capture_profile()}
        self.refresh_window_profiles()
        self.save_preferences()
        self.window_status.setText("Профиль сохранён для выбранного окна")

    def apply_named_profile(self, key):
        entry = self.window_profiles.get(key)
        if not isinstance(entry, dict) or "profile" not in entry:
            return
        try:
            self.apply_profile(entry["profile"])
            self.window_status.setText("Применён профиль: " + str(entry.get("name", key)))
        except (ValueError, TypeError):
            self.window_status.setText("Сохранённый профиль повреждён; текущие настройки сохранены")

    def load_window_profile(self, *_):
        key = self.window_profile_combo.currentData()
        if key is not None:
            self.apply_named_profile(key)

    def remove_window_profile(self):
        key = self.window_profile_combo.currentData()
        if key is not None:
            self.window_profiles.pop(key, None)
            self.refresh_window_profiles()
            self.save_preferences()

    def start_export(self):
        if self.active_source == "photo":
            self.save_photo_dialog()
            return
        if self.studio.batch.running:
            self.export_status.setText("Сначала остановите очередь экспорта")
            return
        if self.export_busy:
            return
        if self.active_source != "file":
            self.export_status.setText(
                "Для живого источника используйте «Запись выходного потока». Экспорт обрабатывает видеофайлы."
            )
            return
        if self.video_path is None:
            self.export_status.setText("Сначала откройте видео")
            return
        scene_mode = bool(self.scene_slots)
        object_mode = self.settings.subject_mode == "object" and not scene_mode
        cached = self.settings.automation.cached(
            "single", self.settings.subject_mode, self.latest_media_time
        )[0]
        if object_mode and self.object_seed is None and not cached:
            self.export_status.setText("Сначала выделите предмет мышкой")
            return
        dialog = ExportDialog(object_mode and not cached, self, scene_mode=scene_mode)
        if cached:
            dialog.start.setCurrentIndex(1)
        dialog.fps.setCurrentIndex(self.fps_combo.currentIndex())
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить обработанное видео",
            str(self.video_path.with_name(self.video_path.stem + "-framed.mp4")),
            "MP4 (*.mp4)",
        )
        if not destination:
            return
        start = (
            self.object_seed[1]
            if object_mode and self.object_seed is not None and not cached
            else (self.latest_media_time if dialog.start.currentIndex() else 0)
        )
        selection = (
            self.object_seed[0]
            if object_mode and self.object_seed is not None and not cached
            else (
                self.input_view.boxes[0].center
                if dialog.start.currentIndex() and self.input_view.boxes
                else None
            )
        )
        if scene_mode:
            start = self.last_render_time
            selection = self.current_selections()
        self.begin_export(
            ExportJob(
                self.video_path,
                Path(destination),
                self.settings,
                [30, 24, 15][dialog.fps.currentIndex()],
                start,
                dialog.audio.isChecked(),
                selection,
            )
        )

    def begin_export(self, job):
        if self.export_busy:
            return
        if not job.frozen:
            try:
                self.studio.validate_analysis_source()
            except (ValueError, OSError) as error:
                self.export_status.setText(str(error))
                return False
            job = replace(job, settings=freeze_framing(job.settings, self.last_tracks))
        if job.source.resolve() == job.destination.resolve():
            self.export_status.setText("Выберите другой файл: исходное видео нельзя перезаписывать")
            return
        self.export_worker.cancel.clear()
        self.export_result = None
        self.export_busy = True
        self.export_button.setEnabled(False)
        self.export_cancel.setEnabled(True)
        self.export_progress.setValue(0)
        self.export_status.setText("Экспорт… Настройки зафиксированы на момент запуска.")
        self.jobs.export.emit(job)
        return True

    def export_finished(self, result):
        self.export_result = result
        self.export_busy = False
        self.export_button.setEnabled(True)
        self.export_cancel.setEnabled(False)
        self.export_status.setText(
            "Экспорт отменён; исходный файл результата сохранён"
            if result["cancelled"]
            else f"Сохранено {result['frames']} кадров: {result['path']}"
        )
        self.studio.batch.finished(result)

    def export_failed(self, message):
        self.export_finished({"cancelled": True, "error": message})
        self.export_status.setText("Ошибка экспорта: " + message)

    def capture_profile(self):
        return {
            "version": 9,
            "surface_lock": self.settings.surface_lock,
            "frame_mask": self.settings.frame_mask.record(),
            "framing_zoom": self.settings.framing_zoom,
            "padding": self.padding.value(),
            "response": self.smoothing.value(),
            "fill": self.fit_combo.currentIndex(),
            "loss": self.loss.currentIndex(),
            "mapping": self.mode.currentIndex(),
            "resolution": self.resolution.currentIndex(),
            "fps": self.fps_combo.currentIndex(),
            "auto_aspect": self.auto_aspect.isChecked(),
            "sensitivity": self.sensitivity.currentData(),
            "subject_mode": self.subject_mode.currentData(),
            "align_eyes": self.align_eyes.isChecked(),
            "eye_level": self.eye_level.value(),
            "max_zoom": self.max_zoom.value(),
            "subject_region": [
                self.settings.subject_region.x,
                self.settings.subject_region.y,
                self.settings.subject_region.w,
                self.settings.subject_region.h,
            ],
            "motion_mode": self.motion_mode.currentData(),
            "framing": self.settings.framing.record(),
            "slots": [slot.record() for slot in self.scene_slots],
            "active_slot": self.active_scene_slot,
            "remove_background": self.remove_background.isChecked(),
            "segmentation_quality": self.segmentation_quality.currentData(),
            "background_color": self.settings.background_color,
            "background_image": self.settings.background_image,
            "calibration": {
                "crop": [
                    self.settings.calibration.crop.x,
                    self.settings.calibration.crop.y,
                    self.settings.calibration.crop.w,
                    self.settings.calibration.crop.h,
                ],
                "mirror": self.settings.calibration.mirror,
                "zoom": self.settings.calibration.zoom,
                "offset_x": self.settings.calibration.offset_x,
                "offset_y": self.settings.calibration.offset_y,
                "rotation": self.settings.calibration.rotation,
            },
            "roi": [self.roi.x, self.roi.y, self.roi.w, self.roi.h] if self.roi else None,
            "target": [
                self.settings.target.x,
                self.settings.target.y,
                self.settings.target.w,
                self.settings.target.h,
            ],
        }

    def apply_profile(self, data):
        profile = validate_profile(data)
        if sys.platform == "win32":
            # The first Windows port supports faces/selected objects; Apple-only modes are disabled.
            profile["remove_background"] = False
            if profile["subject_mode"] not in ("face", "object"):
                profile["subject_mode"] = "face"
            profile["slots"] = [
                dict(
                    slot,
                    subject_mode=slot["subject_mode"]
                    if slot["subject_mode"] in ("face", "object")
                    else "face",
                )
                for slot in profile["slots"]
            ]
        if not self.restoring:
            self.generation += 1
            self.stable = StableTarget()
        previous_restoring, self.restoring = self.restoring, True
        widgets = [
            self.padding,
            self.smoothing,
            self.fit_combo,
            self.loss,
            self.mode,
            self.resolution,
            self.fps_combo,
            self.auto_aspect,
            self.sensitivity,
            self.subject_mode,
            self.align_eyes,
            self.eye_level,
            self.max_zoom,
            self.remove_background,
            self.segmentation_quality,
            self.motion_mode,
            self.subject_region_combo,
            self.auto_framing_enabled,
            *self.framing_controls(),
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.padding.setValue(profile["padding"])
            self.smoothing.setValue(profile["response"])
            for key, combo in [
                ("fill", self.fit_combo),
                ("loss", self.loss),
                ("mapping", self.mode),
                ("resolution", self.resolution),
                ("fps", self.fps_combo),
            ]:
                combo.setCurrentIndex(profile[key])
            self.auto_aspect.setChecked(profile["auto_aspect"])
            self.sensitivity.setCurrentIndex(self.sensitivity.findData(profile["sensitivity"]))
            self.subject_mode.setCurrentIndex(self.subject_mode.findData(profile["subject_mode"]))
            self.align_eyes.setChecked(profile["align_eyes"])
            self.eye_level.setValue(profile["eye_level"])
            self.max_zoom.setValue(profile["max_zoom"])
            self.motion_mode.setCurrentIndex(self.motion_mode.findData(profile["motion_mode"]))
            self.load_framing_controls(FramingRule.from_record(profile["framing"]))
            self.remove_background.setChecked(profile["remove_background"])
            self.segmentation_quality.setCurrentIndex(
                self.segmentation_quality.findData(profile["segmentation_quality"])
            )
            self.scene_slots = [SceneSlot.from_record(value) for value in profile["slots"]]
            self.active_scene_slot = profile["active_slot"] or (
                self.scene_slots[0].identifier if self.scene_slots else ""
            )
            self.slot_stabilizers = {}
            self.settings = replace(
                self.settings,
                background_color=profile["background_color"],
                background_image=profile["background_image"],
                subject_region=Rect(*profile["subject_region"]),
                frame_mask=FrameMask.from_record(profile["frame_mask"]),
                framing_zoom=profile["framing_zoom"],
                surface_lock=profile["surface_lock"],
            )
            self.studio.sync_controls(self.settings)
            self.sync_region_controls()
            if self.scene_slots:
                self.mode.setCurrentIndex(1)
            calibration = profile["calibration"]
            self.settings = replace(
                self.settings,
                calibration=Calibration(
                    Rect(*calibration["crop"]),
                    calibration["mirror"],
                    calibration["zoom"],
                    calibration["offset_x"],
                    calibration["offset_y"],
                    calibration["rotation"],
                ),
            )
            self.roi = Rect(*profile["roi"]) if profile.get("roi") else None
            if profile.get("target"):
                self.settings = replace(self.settings, target=Rect(*profile["target"]))
            self.read_settings(persist=False)
            self.refresh_scene_combo()
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self.restoring = previous_restoring
        if not self.restoring:
            self.invalidate_frame(clear=False)
            self.request_render()
            self.save_timer.start(800)
            self.queue_obs_sync()

    def restore_preferences(self):
        if self.saved_state.get("profile"):
            self.apply_profile(self.saved_state["profile"])
        geometry = self.saved_state.get("geometry")
        if isinstance(geometry, str):
            try:
                self.restoreGeometry(bytes.fromhex(geometry))
            except ValueError:
                pass

    def save_preferences(self):
        if self.restoring:
            return
        try:
            self.preferences.save(
                {
                    "profile": self.capture_profile(),
                    "output_port": self.preferred_port,
                    "local_output_enabled": self.publisher.url is not None,
                    "last_directory": self.last_directory,
                    "geometry": bytes(self.saveGeometry()).hex(),
                    "window_profiles": self.window_profiles,
                    "scenarios": self.scenarios,
                    "connection_profile": self.connection_profile,
                    "signal_protection": self.signal_policy.record(),
                }
            )
        except OSError:
            self.status.setText("Не удалось сохранить настройки. Текущий сеанс продолжает работать.")

    def save_profile(self):
        file, _ = QFileDialog.getSaveFileName(self, "Сохранить настройки", "framepilot.json", "JSON (*.json)")
        if file:
            try:
                Path(file).write_text(
                    json.dumps(self.capture_profile(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError as error:
                QMessageBox.warning(self, "Профиль", str(error))

    def load_profile(self):
        file, _ = QFileDialog.getOpenFileName(self, "Загрузить настройки", "", "JSON (*.json)")
        if not file:
            return
        try:
            self.apply_profile(json.loads(Path(file).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "Профиль", str(error))

    def drain_processing_for_close(self):
        self.flush_render()
        deadline = time.monotonic() + 10
        while self.busy:
            receipt = {"ready": threading.Event()}
            self.jobs.drain_video.emit(receipt)
            if not receipt["ready"].wait(max(0, deadline - time.monotonic())):
                return False
            # Consume the serial worker result without a nested Qt event loop during close.
            if receipt["error"] is not None:
                self.video_error(*receipt["error"])
            elif receipt["result"] is not None:
                self.frame_ready(receipt["result"])
            else:
                return False
        return not (self.render_pending and self.latest_frame is not None)

    def closeEvent(self, event: QCloseEvent):
        name_dialog = getattr(self, "camera_name_dialog", None)
        if name_dialog is not None:
            if name_dialog.busy:
                name_dialog.raise_()
                name_dialog.reject()
                event.ignore()
                return
            name_dialog.reject()
        if self.connection_wizard is not None:
            wizard = self.connection_wizard
            wizard.reject()
            if wizard.thread.isRunning():
                QTimer.singleShot(500, self.close)
                event.ignore()
                return
        if not self.closing_requested:
            self.closing_requested = True
            self.pause_source()
            self.camera_input.stop()
            self.animation_timer.stop()
            self.scan_timer.stop()
            self.obs_timer.stop()
            self.studio.worker.cancel.set()
            self.studio.recorder.stop()
            self.export_worker.cancel.set()
            self.obs_worker.cancel.set()
        if not self.closing and not self.drain_processing_for_close():
            self.status.setText("Ожидание последнего кадра перед сохранением сеанса…")
            QTimer.singleShot(250, self.close)
            event.ignore()
            return
        self.closing = True
        if not self.studio.close():
            QTimer.singleShot(500, self.close)
            event.ignore()
            return
        self.save_timer.stop()
        self.animation_timer.stop()
        self.signal_timer.stop()
        self.save_preferences()
        self.scan_timer.stop()
        self.obs_timer.stop()
        self.obs_sync_timer.stop()
        self.player.stop()
        self.camera_input.stop()
        self.publisher.stop()
        self.obs_worker.cancel.set()
        self.export_worker.cancel.set()
        self.video_thread.quit()
        self.capture_thread.quit()
        self.obs_thread.quit()
        self.export_thread.quit()
        # Capture has bounded timeouts; never terminate a thread inside an Apple API.
        if not all(
            [
                self.video_thread.wait(12000),
                self.capture_thread.wait(12000),
                self.obs_thread.wait(5000),
                self.export_thread.wait(5000),
            ]
        ):
            self.status.setText("Ожидание завершения обработки…")
            QTimer.singleShot(500, self.close)
            event.ignore()
            return
        self.output_window.close()
        self.demo.timer.stop()
        self.demo.hide()
        if self.sample_directory is not None:
            self.sample_directory.cleanup()
            self.sample_directory = None
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="FramePilot — local video framing")
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--demo", action="store_true", help="Open the changing test target")
    parser.add_argument(
        "--sample", action="store_true", help="Generate and play the included NASA test portrait"
    )
    parser.add_argument("--obs", action="store_true", help="Start local OBS browser output")
    parser.add_argument("--camera-name-task", help=argparse.SUPPRESS)
    parser.add_argument(
        "--portable-self-test",
        action="store_true",
        help="Check files, photos, UI and MP4 on Windows or macOS",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Check the packaged app without external permissions"
    )
    parser.add_argument("--report", type=Path, help="Path for the self-test JSON report")
    parser.add_argument(
        "--tracking-check",
        action="store_true",
        help="Run deterministic tracking acceptance cases without a camera or OBS",
    )
    parser.add_argument(
        "--obs-self-test",
        action="store_true",
        help="Explicitly test a local OBS Virtual Camera; requires idle OBS with WebSocket enabled",
    )
    args = parser.parse_args()
    if args.camera_name_task is not None:
        from .windows_camera_names import run_task

        sys.exit(run_task(args.camera_name_task))
    if args.tracking_check:
        from . import __version__
        from .tracking_validation import validate_tracking

        result = {"version": __version__, **validate_tracking()}
        report = args.report or Path.cwd() / "framepilot-tracking-check.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.exit(0 if result["passed"] else 1)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("FramePilot")
    install_appearance(app)
    if args.portable_self_test or (args.self_test and sys.platform != "darwin"):
        from .portable_diagnostics import portable_smoke_test

        sys.exit(portable_smoke_test(app, args.report or Path.cwd() / "framepilot-portable-check.json"))
    if args.obs_self_test:
        from .connection_diagnostics import obs_self_test

        sys.exit(obs_self_test(app, args.report or Path.cwd() / "framepilot-obs-check.json"))
    if args.self_test:
        from .diagnostics import smoke_test

        sys.exit(smoke_test(app, args.report or Path.cwd() / "framepilot-check.json"))
    window = MainWindow()
    window.show()
    if args.demo:
        window.show_demo()
    if args.video:
        window.open_media(args.video)
    elif args.sample:
        window.load_sample()
    if args.obs:
        window.start_local_output()
    sys.exit(app.exec())
