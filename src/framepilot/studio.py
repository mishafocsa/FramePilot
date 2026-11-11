"""Clip tools and live-source controls, kept separate from the main playback controller."""

import json
import sys
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .analysis import AnalysisJob, AnalysisWorker
from .automation import Automation, Cue, source_identity
from .batch import BatchExport
from .masks import FrameMask, read_alpha
from .media_check import MediaCheck
from .object_geometry import QuadSelection
from .preferences import Preferences, decode_selections, encode_selections, validate_profile
from .recording import LiveRecorder
from .recovery import RecoveryStore
from .screen_source import create_screen_input
from .widgets import to_array


def clock(seconds):
    return f"{int(seconds) // 60:02}:{seconds % 60:05.2f}"


def note(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("mutedText")
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return label


class ClipMarkers(QWidget):
    seek = Signal(int)

    def __init__(self):
        super().__init__()
        self.duration = 0
        self.cues = ()
        self.problems = []
        self.setFixedHeight(20)
        self.setToolTip(
            "Мятные точки — поправки; жёлтые участки — потеря или слабое отслеживание. Нажмите для перехода."
        )
        self.setAccessibleName("Отметки анализа и поправок ролика")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#2a3a4d"), 2))
        painter.drawLine(4, 10, max(4, self.width() - 4), 10)
        if self.duration <= 0:
            return
        scale = (self.width() - 8) / self.duration
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#dfa84c"))
        for start, end, _ in self.problems:
            painter.drawRoundedRect(QRectF(4 + start * scale, 7, max(3, (end - start) * scale), 6), 3, 3)
        painter.setBrush(QColor("#79e7c9"))
        for cue in self.cues:
            painter.drawEllipse(QPointF(4 + cue.time * scale, 10), 3, 3)

    def mousePressEvent(self, event):
        if self.duration > 0 and event.button() == Qt.MouseButton.LeftButton:
            self.seek.emit(
                round(
                    max(0, min(1, (event.position().x() - 4) / max(1, self.width() - 8)))
                    * self.duration
                    * 1000
                )
            )


class Studio(QObject):
    analyze = Signal(object)

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.screen = create_screen_input(self)
        self.screen.frame.connect(self.screen_frame)
        self.screen.sources.connect(self.sources_ready)
        self.screen.message.connect(self.screen_message)
        self.recorder = LiveRecorder()
        self.recording_seen = False
        self.revision = 0
        self.analysis_busy = False
        self.analysis_identity = None
        self.mode_signature = None
        self.media = MediaCheck(self)
        self.recovery = RecoveryStore(main.preferences.path.with_name("session-recovery.json"), self)
        self.recovery_data = self.recovery_pending = None
        self.pause_after_recovery = False
        self.pending_photo_state = None
        self.recovery.available.connect(self.recovery_available)
        self.recovery.load_request.emit(self.recovery.path)
        self.batch = BatchExport(self)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self.autosave)
        self.periodic_save = QTimer(self)
        self.periodic_save.timeout.connect(self.autosave)
        self.periodic_save.start(30000)
        self.analysis_thread = QThread(self)
        self.worker = AnalysisWorker()
        self.worker.moveToThread(self.analysis_thread)
        self.analyze.connect(self.worker.run)
        self.worker.completed.connect(self.analyzed)
        self.worker.failed.connect(self.analysis_failed)
        self.analysis_thread.finished.connect(self.worker.deleteLater)
        self.analysis_thread.start()
        self.markers = ClipMarkers()
        self.markers.seek.connect(main.seek_to)
        main.player.durationChanged.connect(self.update_markers)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_recording)
        self.poll_timer.start(200)

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def groups(self):
        result = []
        group = QGroupBox("Фото")
        layout = QVBoxLayout(group)
        self.photo_status = note(
            "JPEG/PNG: выберите лицо или выделите предмет на исходнике. Прозрачные области показывают фон композиции."
        )
        layout.addWidget(self.photo_status)
        self.button("Показать всё фото", self.main.show_whole_photo, layout)
        self.button("Найти лицо на фото", self.main.photo_face, layout)
        self.button("Сохранить PNG / JPEG…", self.main.save_photo_dialog, layout)
        result.append(group)
        group = QGroupBox("Совместимость видео")
        layout = QVBoxLayout(group)
        self.media_status = note(
            "Расширение не определяет поддержку: проверяются первый видеокадр FFmpeg и фактический предпросмотр Qt."
        )
        layout.addWidget(self.media_status)
        self.media.changed.connect(self.media_status.setText)
        self.button("Проверить открытый ролик", self.inspect_media, layout)
        result.append(group)
        group = QGroupBox("Восстановление сеанса")
        layout = QVBoxLayout(group)
        self.recovery_status = note(
            "Файловый сеанс сохраняется автоматически. Камера и захват экрана при восстановлении не запускаются."
        )
        layout.addWidget(self.recovery_status)
        self.recovery.status.connect(self.recovery_status.setText)
        self.restore_button = self.button("Восстановить последний сеанс", self.restore_recovery, layout)
        self.restore_button.setEnabled(False)
        result.append(group)
        group = QGroupBox("Захват экрана и окна")
        layout = QVBoxLayout(group)
        self.source_combo = QComboBox()
        self.source_combo.setMinimumContentsLength(10)
        self.source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.source_combo.addItem("Обновите список источников…", None)
        layout.addWidget(self.source_combo)
        self.button("Обновить экраны и окна", self.screen.refresh, layout)
        row = QHBoxLayout()
        self.button("Начать захват", self.start_screen, row)
        self.button("Остановить", self.stop_screen, row)
        layout.addLayout(row)
        layout.addWidget(
            note(
                "Windows: захват выбранного окна через Qt. Полный рабочий стол пока недоступен. Звук не захватывается."
                if sys.platform == "win32"
                else "Нужен доступ к записи экрана. Собственные окна FramePilot исключаются из захвата экрана. Звук источника не захватывается."
            )
        )
        result.append(group)

        group = QGroupBox("Анализ ролика")
        layout = QVBoxLayout(group)
        self.analysis_button = self.button("Анализировать от текущего кадра", self.start_analysis, layout)
        self.analysis_cancel = self.button("Отменить анализ", self.cancel_analysis, layout)
        self.analysis_cancel.setEnabled(False)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.worker.progress.connect(self.progress.setValue)
        self.analysis_status = note(
            "Выберите цель. Анализ идёт от текущего кадра до конца; предыдущая часть остаётся непроверенной."
        )
        layout.addWidget(self.analysis_status)
        self.use_analysis = QCheckBox("Применять анализ и поправки")
        self.use_analysis.setChecked(True)
        self.use_analysis.toggled.connect(self.toggle_automation)
        layout.addWidget(self.use_analysis)
        self.problems = QListWidget()
        self.problems.setMaximumHeight(150)
        self.problems.itemActivated.connect(self.seek_problem)
        self.problems.itemClicked.connect(self.seek_problem)
        layout.addWidget(self.problems)
        self.button("Сбросить результаты анализа", self.clear_analysis, layout)
        result.append(group)

        group = QGroupBox("Поправки по времени")
        layout = QVBoxLayout(group)
        layout.addWidget(
            note(
                "На паузе выберите нужную цель, часть и масштаб, затем сохраните поправку. Повторите анализ после изменения выбора; он обеспечит точную перемотку по рассчитанной траектории."
            )
        )
        self.cue_list = QListWidget()
        self.cue_list.setMaximumHeight(140)
        self.cue_list.itemActivated.connect(self.seek_cue)
        layout.addWidget(self.cue_list)
        self.button("Добавить / заменить поправку", self.add_cue, layout)
        self.button("Удалить выбранную поправку", self.remove_cue, layout)
        self.button("Сохранить проект ролика…", self.save_project_dialog, layout)
        self.button("Открыть проект ролика…", self.load_project_dialog, layout)
        self.cue_status = note("Поправки относятся только к открытому ролику и выбранной рамке.")
        layout.addWidget(self.cue_status)
        result.append(group)

        group = QGroupBox("Маска и форма рамки")
        layout = QVBoxLayout(group)
        self.mask_kind = QComboBox()
        for title, kind in (
            ("По найденной рамке", "auto"),
            ("Прямоугольник", "rectangle"),
            ("Овал", "ellipse"),
            ("Круг", "circle"),
            ("Скруглённые углы", "rounded"),
            ("Маска из PNG", "png"),
        ):
            self.mask_kind.addItem(title, kind)
        self.mask_kind.currentIndexChanged.connect(self.mask_changed)
        layout.addWidget(self.mask_kind)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0, 0.5)
        self.radius.setSingleStep(0.05)
        self.radius.setValue(0.15)
        self.radius.valueChanged.connect(self.mask_changed)
        form = QFormLayout()
        form.addRow("Скругление (0–0,5)", self.radius)
        layout.addLayout(form)
        self.mask_path = ""
        self.button("Выбрать PNG с прозрачностью…", self.choose_mask, layout)
        self.mask_status = note(
            "Альфа-канал задаёт видимую область. Цвет PNG игнорируется. Маска действует внутри активной рамки."
        )
        layout.addWidget(self.mask_status)
        result.append(group)

        group = QGroupBox("Запись выходного потока")
        layout = QVBoxLayout(group)
        self.record_button = self.button("Начать запись MP4…", self.record_dialog, layout)
        self.record_stop = self.button("Остановить и сохранить", self.stop_recording, layout)
        self.record_stop.setEnabled(False)
        self.record_status = note(
            "Записывается фактический выход, включая чёрный кадр. MP4 H.264 без звука; разрешение и FPS фиксируются при старте."
        )
        layout.addWidget(self.record_status)
        result.append(group)

        group = QGroupBox("Качество и поворот")
        layout = QVBoxLayout(group)
        self.surface_lock = QCheckBox("Часть следует за поворотом")
        self.surface_lock.toggled.connect(self.main.read_settings)
        layout.addWidget(self.surface_lock)
        layout.addWidget(
            note(
                "Для выделенного предмета: часть привязана к исходной поверхности при повороте в плоскости изображения. Поворот в 3D и деформация не гарантируются."
            )
        )
        self.composition_status = note("Показатели появятся после выбора цели.")
        layout.addWidget(self.composition_status)
        self.button("Вписать и сбросить приближение", self.fit_reset, layout)
        result.append(group)
        result.append(self.batch.group())
        return result

    def inspect_media(self):
        if self.main.active_source == "file" and self.main.video_path is not None:
            self.media.start(self.main.video_path)

    def schedule_save(self):
        if not self.main.restoring and not self.main.closing:
            self.autosave_timer.start(1500)

    def snapshot(self):
        main = self.main
        source = main.photo_path if main.active_source == "photo" else main.video_path
        if (
            main.active_source not in ("file", "photo")
            or source is None
            or main.latest_frame is None
            or main.busy
        ):
            return None
        return {
            "source": str(source),
            "source_kind": main.active_source,
            "position": max(0, round(main.last_render_time * 1000)),
            "profile": main.capture_profile(),
            "automation": main.settings.automation,
            "selections": encode_selections(main.current_selections()),
            "expected_identity": main.photo_input.data.identity
            if main.active_source == "photo" and main.photo_input.data is not None
            else self.analysis_identity
            or (
                self.media.result["identity"]
                if self.media.result and self.media.path == main.video_path
                else None
            ),
        }

    def autosave(self):
        self.recovery.save(self.snapshot())

    def recovery_available(self, answer):
        data, error = answer
        self.recovery_data = data
        self.restore_button.setEnabled(data is not None)
        if data:
            self.recovery_status.setText("Доступен сеанс: " + Path(data["source"]).name)
        elif error:
            self.recovery_status.setText("Сеанс не восстановлен: " + error)

    def restore_recovery(self):
        data = self.recovery_data
        if data is None:
            return
        try:
            if source_identity(data["source"]) != data["identity"]:
                raise ValueError("Исходный ролик изменился")
            main = self.main
            if data.get("source_kind", "file") == "photo":
                main.load_photo(Path(data["source"]))
                self.pending_photo_state = data
                self.recovery_status.setText("Восстанавливаю фото…")
                return
            main.load_video(Path(data["source"]))
            main.apply_profile(data["profile"])
            self.apply_automation(data["automation"], preserve_override=True)
            self.analysis_identity = data["identity"] if data["automation"].tracks else None
            self.recovery_pending = data
            if main.player.duration() > 0:
                self.duration_ready(main.player.duration())
            self.recovery_status.setText("Восстанавливаю файловый сеанс…")
        except (ValueError, OSError, TypeError, KeyError) as error:
            self.recovery_status.setText(str(error))

    def restore_photo_state(self):
        data, self.pending_photo_state = self.pending_photo_state, None
        if data is None or self.main.photo_path != Path(data["source"]):
            return
        main = self.main
        main.apply_profile(data["profile"])
        selections = decode_selections(data.get("selections", {}))
        main.click = selections if main.scene_slots else selections.get("single", main.click)
        main.object_seeds = {
            key: (value, 0)
            for key, value in selections.items()
            if hasattr(value, "w") or isinstance(value, QuadSelection)
        }
        main.object_seed = main.object_seeds.get(main.active_scene_slot or "single")
        main.reset_pending = True
        self.recovery_status.setText("Фото и настройки восстановлены")

    def duration_ready(self, duration):
        data = self.recovery_pending
        if data is None or duration <= 0 or self.main.video_path != Path(data["source"]):
            return
        self.recovery_pending = None
        main = self.main
        position = min(data["position"], max(0, duration - 50))
        main.seek_to(position)
        selections = decode_selections(data.get("selections", {}))
        main.click = selections if main.scene_slots else selections.get("single")
        main.object_seeds = {
            key: (value, position / 1000)
            for key, value in selections.items()
            if hasattr(value, "w") or isinstance(value, QuadSelection)
        }
        main.object_seed = main.object_seeds.get(main.active_scene_slot or "single")
        self.pause_after_recovery = True
        if main.latest_frame is not None and abs(main.latest_media_time - position / 1000) < 0.1:
            self.recovery_frame_ready()
            main.request_render()

    def recovery_frame_ready(self):
        if self.pause_after_recovery:
            self.pause_after_recovery = False
            self.main.player.pause()
            self.recovery_status.setText("Сеанс восстановлен на паузе")

    def fit_reset(self):
        main = self.main
        main.fit_combo.setCurrentIndex(0)
        self.zoom.setValue(1)
        main.settings = replace(
            main.settings, calibration=replace(main.settings.calibration, zoom=1, offset_x=0, offset_y=0)
        )
        main.read_settings()

    def update_composition(self, report):
        self.main.last_composition = report
        lines = []
        if "visible_fraction" in report:
            lines.append(f"Видимая выбранная область: ≈ {report['visible_fraction'] * 100:.0f}%")
        if "upscale" in report:
            lines.append(f"Масштаб исходных пикселей: {report['upscale']:.2f}×")
        if report.get("surface_locked"):
            lines.append("Привязка к поверхности активна")
        lines.extend(report.get("warnings", []))
        lines.append("Оценка по прямоугольнику/полигону, без сегментации предмета.")
        self.composition_status.setText("\n".join(lines))

    def zoom_control(self):
        self.zoom = QDoubleSpinBox()
        self.zoom.setRange(0.5, 2)
        self.zoom.setSingleStep(0.1)
        self.zoom.setValue(1)
        self.zoom.setSuffix(" ×")
        self.zoom.valueChanged.connect(self.main.read_settings)
        return self.zoom

    def mask_value(self):
        kind = self.mask_kind.currentData()
        if kind == "png" and not self.mask_path:
            return FrameMask()
        return FrameMask(kind, self.radius.value(), self.mask_path)

    def sync_controls(self, settings):
        for control, value in (
            (self.mask_kind, self.mask_kind.findData(settings.frame_mask.kind)),
            (self.radius, settings.frame_mask.radius),
            (self.zoom, settings.framing_zoom),
        ):
            blocked = control.blockSignals(True)
            if isinstance(control, QComboBox):
                control.setCurrentIndex(value)
            else:
                control.setValue(value)
            control.blockSignals(blocked)
        self.mask_path = settings.frame_mask.image
        blocked = self.surface_lock.blockSignals(True)
        self.surface_lock.setChecked(settings.surface_lock)
        self.surface_lock.blockSignals(blocked)

    def mask_changed(self, *_):
        if self.mask_kind.currentData() == "png" and not self.mask_path:
            self.mask_status.setText("Выберите PNG с альфа-каналом; до этого применяется найденная форма.")
        self.main.read_settings()

    def choose_mask(self):
        file, _ = QFileDialog.getOpenFileName(self.main, "Маска рамки", "", "PNG (*.png)")
        if not file:
            return
        try:
            read_alpha(file)
            self.mask_path = str(Path(file).resolve())
            self.mask_kind.setCurrentIndex(self.mask_kind.findData("png"))
            self.mask_changed()
            self.mask_status.setText(Path(file).name)
        except (ValueError, OSError) as error:
            self.mask_status.setText(str(error))

    def configuration(self):
        return tuple((s.identifier, s.subject_mode) for s in self.main.settings.slots) or (
            ("single", self.main.settings.subject_mode),
        )

    def settings_changed(self):
        configuration = self.configuration()
        if self.mode_signature is not None and configuration != self.mode_signature:
            self.selection_changed()
            previous = dict(self.mode_signature)
            unchanged = {identifier for identifier, mode in configuration if previous.get(identifier) == mode}
            self.main.settings = replace(
                self.main.settings,
                automation=replace(
                    self.main.settings.automation,
                    cues=tuple(c for c in self.main.settings.automation.cues if c.slot in unchanged),
                ),
            )
            self.update_markers()
        self.mode_signature = configuration
        self.schedule_save()

    def selection_changed(self):
        self.revision += 1
        self.worker.cancel.set()
        self.analysis_identity = None
        self.main.settings = replace(
            self.main.settings, automation=replace(self.main.settings.automation, tracks=())
        )
        if hasattr(self, "analysis_status"):
            self.analysis_status.setText("Выбор цели изменён. Повторите анализ ролика.")
            self.update_markers()

    def new_source(self):
        self.pending_photo_state = None
        self.recovery_pending = None
        self.pause_after_recovery = False
        self.main.last_composition = {}
        self.main.input_view.polygons = []
        self.selection_changed()
        self.main.settings = replace(
            self.main.settings, automation=Automation(enabled=self.use_analysis.isChecked())
        )
        self.progress.setValue(0)
        self.update_markers()

    def apply_automation(self, automation, preserve_override=False):
        self.main.settings = replace(
            self.main.settings,
            automation=automation if preserve_override else replace(automation, override_time=None),
        )
        self.update_markers()
        self.main.invalidate_frame(clear=False)
        self.main.request_render()
        self.schedule_save()

    def toggle_automation(self, enabled):
        self.apply_automation(replace(self.main.settings.automation, enabled=enabled))

    def update_markers(self, *_):
        automation = self.main.settings.automation
        slot = self.main.active_scene_slot or "single"
        self.markers.duration = self.main.player.duration() / 1000 if self.main.active_source == "file" else 0
        self.markers.cues = automation.keys(slot)
        self.markers.problems = [
            span for track in automation.tracks if track.slot == slot for span in track.problems()
        ]
        self.markers.update()
        self.cue_list.clear()
        for cue in automation.keys(slot):
            item = QListWidgetItem(f"{clock(cue.time)} · масштаб {cue.zoom:.2f}×")
            item.setData(Qt.ItemDataRole.UserRole, cue)
            self.cue_list.addItem(item)
        self.problems.clear()
        names = {
            "lost": "Цель не подтверждена",
            "weak": "Слабые ориентиры",
            "ambiguous": "Похожие кандидаты",
            "recovering": "Восстановление",
        }
        for start, end, state in self.markers.problems[:500]:
            item = QListWidgetItem(f"{clock(start)}–{clock(end)} · {names.get(state, state)}")
            item.setData(Qt.ItemDataRole.UserRole, start)
            self.problems.addItem(item)

    def add_cue(self):
        main = self.main
        if main.active_source != "file" or main.video_path is None or main.latest_frame is None or main.busy:
            self.cue_status.setText("Откройте ролик, дождитесь обработки и поставьте на паузу.")
            return
        main.pause_source()
        if not main.input_view.boxes:
            self.cue_status.setText("Сначала выберите и подтвердите цель на исходнике.")
            return
        cue = Cue(
            main.latest_media_time,
            main.active_scene_slot or "single",
            main.input_view.boxes[0],
            main.settings.subject_region,
            main.padding.value(),
            self.zoom.value(),
            tuple(
                tuple(float(v) for v in p)
                for p in main.last_tracks.get(main.active_scene_slot or "single", {}).get("quad", [])
            )
            if main.settings.surface_lock
            and main.last_tracks.get(main.active_scene_slot or "single", {}).get("quad") is not None
            else None,
        )
        cues = tuple(
            c
            for c in main.settings.automation.cues
            if c.slot != cue.slot
            or abs(c.time - cue.time) > 0.5 / [30, 24, 15][main.fps_combo.currentIndex()]
        ) + (cue,)
        self.selection_changed()
        self.apply_automation(replace(main.settings.automation, cues=cues, tracks=()))
        self.cue_status.setText(
            "Поправка сохранена. Для точной перемотки повторите анализ с первого нужного кадра."
        )

    def remove_cue(self):
        item = self.cue_list.currentItem()
        if item is not None:
            cue = item.data(Qt.ItemDataRole.UserRole)
            self.selection_changed()
            self.apply_automation(
                replace(
                    self.main.settings.automation,
                    cues=tuple(c for c in self.main.settings.automation.cues if c != cue),
                    tracks=(),
                )
            )

    def seek_cue(self, item):
        self.main.seek_to(round(item.data(Qt.ItemDataRole.UserRole).time * 1000))

    def seek_problem(self, item):
        self.main.seek_to(round(item.data(Qt.ItemDataRole.UserRole) * 1000))

    def start_analysis(self):
        main = self.main
        if self.analysis_busy:
            return
        if main.active_source != "file" or main.video_path is None or main.latest_frame is None or main.busy:
            self.analysis_status.setText("Откройте ролик и дождитесь обработки выбранного кадра.")
            return
        main.pause_source()
        selections = main.current_selections()
        slots = main.settings.slots
        if any(s.identifier not in selections for s in slots) or (not slots and not main.input_view.boxes):
            self.analysis_status.setText("Сначала подтвердите цель в каждой рамке.")
            return
        self.revision += 1
        self.worker.cancel.clear()
        self.analysis_busy = True
        self.analysis_button.setEnabled(False)
        self.analysis_cancel.setEnabled(True)
        self.progress.setValue(0)
        selection = selections if slots else selections.get("single")
        self.analysis_status.setText("Анализ… Воспроизведение и видеовыход не блокируются.")
        self.analyze.emit(
            AnalysisJob(
                main.video_path,
                main.settings,
                selection,
                main.latest_media_time,
                [30, 24, 15][main.fps_combo.currentIndex()],
                self.revision,
            )
        )

    def cancel_analysis(self):
        self.revision += 1
        self.worker.cancel.set()
        self.analysis_status.setText("Отмена анализа…")

    def analysis_idle(self):
        self.analysis_busy = False
        self.analysis_button.setEnabled(True)
        self.analysis_cancel.setEnabled(False)

    def analyzed(self, result):
        job, identity, tracks = result
        self.analysis_idle()
        if (
            job.token != self.revision
            or self.main.closing_requested
            or self.main.active_source != "file"
            or self.main.video_path != job.source
        ):
            self.analysis_status.setText("Результат устарел после изменения источника или цели.")
            return
        self.analysis_identity = identity
        self.apply_automation(replace(self.main.settings.automation, tracks=tracks))
        problems = sum(len(t.problems()) for t in tracks)
        self.analysis_status.setText(
            f"Готово: {len(tracks)} целей, {sum(len(t.samples) for t in tracks)} кадров целей. Участков с замечаниями: {problems}. Это признаки трекера, не гарантия распознавания."
        )

    def analysis_failed(self, result):
        self.analysis_idle()
        if result[0] == self.revision:
            self.analysis_status.setText(result[1])

    def clear_analysis(self):
        self.selection_changed()
        self.apply_automation(replace(self.main.settings.automation, tracks=()))

    def save_project(self, path):
        main = self.main
        if main.active_source != "file" or main.video_path is None:
            raise ValueError("Сначала откройте ролик")
        identity = source_identity(main.video_path)
        if self.analysis_identity is not None and identity != self.analysis_identity:
            raise ValueError("Ролик изменился; повторите анализ")
        data = {
            "format": "framepilot-clip",
            "version": 1,
            "source_name": main.video_path.name,
            "source_identity": identity,
            "profile": main.capture_profile(),
            "automation": main.settings.automation.record(),
        }
        path = Path(path)
        if path.resolve() == main.video_path.resolve():
            raise ValueError("Нельзя заменить исходный ролик проектом")
        Preferences(path).save(data, compact=True)

    def validate_analysis_source(self):
        if (
            self.main.settings.automation.tracks
            and self.analysis_identity is not None
            and self.main.video_path is not None
            and source_identity(self.main.video_path) != self.analysis_identity
        ):
            raise ValueError("Исходный ролик изменился. Повторите анализ перед экспортом.")

    def load_project(self, path):
        main = self.main
        if main.active_source != "file" or main.video_path is None:
            raise ValueError("Сначала откройте соответствующий ролик")
        if Path(path).stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Проект слишком большой")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("format") != "framepilot-clip"
            or data.get("version") != 1
            or data.get("source_identity") != source_identity(main.video_path)
        ):
            raise ValueError("Проект относится к другому или изменённому ролику")
        profile = validate_profile(data.get("profile"))
        automation = Automation.from_record(data.get("automation"))
        ids = {slot["id"] for slot in profile["slots"]} or {"single"}
        modes = {slot["id"]: slot["subject_mode"] for slot in profile["slots"]} or {
            "single": profile["subject_mode"]
        }
        if any(c.slot not in ids for c in automation.cues) or any(
            t.slot not in ids for t in automation.tracks
        ):
            raise ValueError("Цель проекта отсутствует в профиле")
        if any(t.mode != modes[t.slot] for t in automation.tracks):
            raise ValueError("Категория анализа отличается от профиля")
        self.selection_changed()
        main.apply_profile(profile)
        self.analysis_identity = data["source_identity"] if automation.tracks else None
        blocked = self.use_analysis.blockSignals(True)
        self.use_analysis.setChecked(automation.enabled)
        self.use_analysis.blockSignals(blocked)
        self.apply_automation(automation)

    def save_project_dialog(self):
        file, _ = QFileDialog.getSaveFileName(
            self.main, "Проект ролика", "clip.framepilot.json", "FramePilot (*.json)"
        )
        if file:
            try:
                self.save_project(file)
                self.cue_status.setText("Проект сохранён: " + Path(file).name)
            except (ValueError, OSError) as error:
                self.cue_status.setText(str(error))

    def load_project_dialog(self):
        file, _ = QFileDialog.getOpenFileName(self.main, "Проект ролика", "", "FramePilot (*.json)")
        if file:
            try:
                self.load_project(file)
                self.cue_status.setText("Проект открыт: " + Path(file).name)
            except (ValueError, OSError, TypeError, KeyError) as error:
                self.cue_status.setText(str(error))

    def sources_ready(self, sources):
        self.source_combo.clear()
        for source in sources:
            self.source_combo.addItem(source.title, source)
        if not sources:
            self.source_combo.addItem("Нет доступных источников", None)

    def start_screen(self):
        source = self.source_combo.currentData()
        if source is None:
            self.main.status.setText("Сначала обновите список и выберите экран или окно")
            return
        main = self.main
        main.photo_input.cancel()
        main.photo_path = None
        main.photo_ready = False
        main.photo_opacity = None
        main.pause_source()
        main.camera_input.stop()
        self.new_source()
        main.player.pause()
        main.active_source = "screen"
        main.latest_frame = None
        main.latest_media_time = 0
        main.reset_pending = True
        main.click = main.object_seed = None
        main.object_seeds = {}
        main.last_tracks = {}
        main.input_view.boxes = []
        main.input_view.focus = []
        main.input_view.recovery = []
        main.input_view.markers = []
        main.pending_scenario = None
        main.pending_object_sample = False
        main.object_sample_active = False
        main.seek_pending = False
        main.animation_timer.stop()
        main.telemetry.reset()
        main.invalidate_frame(clear=True)
        main.arm_signal()
        main.timeline.setEnabled(False)
        main.position_label.setText("LIVE")
        main.filename.setText(source.title)
        self.screen.start(source, [30, 24, 15][main.fps_combo.currentIndex()])
        main.editor.navigate(1)

    def stop_screen(self):
        self.screen.stop()
        if self.main.active_source == "screen":
            self.main.invalidate_frame(clear=False)
            self.main.render_pending = False
            self.main.signal_monitor.fail("source_stopped")
            self.main.watch_signal()

    def screen_frame(self, frame):
        if self.main.active_source == "screen" and not self.main.closing:
            self.main.accept_input(frame[0], frame[1], "screen")

    def screen_message(self, message):
        self.main.status.setText(message)
        if self.main.active_source == "screen" and not self.screen.running and not self.screen.paused:
            self.main.invalidate_frame(clear=False)
            self.main.render_pending = False
            self.main.signal_monitor.fail("source_error")
            self.main.watch_signal()

    def record_dialog(self):
        file, _ = QFileDialog.getSaveFileName(
            self.main, "Запись видеовыхода", "FramePilot-recording.mp4", "MP4 (*.mp4)"
        )
        if file:
            try:
                self.begin_recording(Path(file))
            except (ValueError, OSError) as error:
                self.record_status.setText(str(error))

    def begin_recording(self, path):
        if self.main.last_output.isNull():
            raise ValueError("Сначала запустите источник видео")
        if self.main.video_path is not None and Path(path).resolve() == self.main.video_path.resolve():
            raise ValueError("Нельзя перезаписывать исходный ролик")
        if self.main.photo_path is not None and Path(path).resolve() == self.main.photo_path:
            raise ValueError("Нельзя перезаписывать исходное фото")
        self.recorder.start(
            path, to_array(self.main.last_output), [30, 24, 15][self.main.fps_combo.currentIndex()]
        )
        self.recording_seen = True
        self.record_button.setEnabled(False)
        self.record_stop.setEnabled(True)

    def stop_recording(self):
        self.recorder.stop()
        self.record_stop.setEnabled(False)
        self.record_status.setText("Сохраняю MP4…")

    def poll_recording(self):
        if self.recorder.active:
            self.record_status.setText(
                f"● Запись · {clock(self.recorder.frames / self.recorder.fps)} · {self.recorder.size[0]}×{self.recorder.size[1]} · {self.recorder.fps} FPS · без звука"
            )
        elif self.recording_seen:
            self.recording_seen = False
            self.record_button.setEnabled(True)
            self.record_stop.setEnabled(False)
            self.record_status.setText(
                "Ошибка записи: " + self.recorder.error
                if self.recorder.error
                else "Сохранено: " + str(self.recorder.destination)
            )

    def close(self):
        snapshot = self.snapshot()
        self.worker.cancel.set()
        self.poll_timer.stop()
        self.autosave_timer.stop()
        self.periodic_save.stop()
        self.batch.stop()
        media_closed = self.media.close()
        photo_closed = self.main.photo_input.close()
        recovery_closed = self.recovery.close(snapshot)
        self.recorder.close()
        self.analysis_thread.quit()
        screen_closed = self.screen.close()
        analysis_closed = self.analysis_thread.wait(15000)
        return (
            screen_closed
            and analysis_closed
            and media_closed
            and recovery_closed
            and photo_closed
            and not self.recorder.active
        )
