"""Sequential exports with an immutable recipe and source identity for each clip."""

import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from .automation import source_identity
from .exporter import ExportJob
from .framing import freeze_framing


@dataclass
class QueueEntry:
    job: ExportJob
    state: str = "pending"
    error: str = ""


class BatchExport(QObject):
    def __init__(self, studio):
        super().__init__(studio)
        self.studio = studio
        self.main = studio.main
        self.entries = []
        self.running = False
        self.current = None

    def group(self):
        group = QGroupBox("Очередь экспорта")
        layout = QVBoxLayout(group)
        self.list = QListWidget()
        self.list.setMaximumHeight(180)
        layout.addWidget(self.list)
        for text, callback in (
            ("Добавить от текущего кадра…", self.add_current),
            ("Запустить очередь", self.start),
            ("Остановить очередь", self.stop),
            ("Удалить выбранное", self.remove),
            ("Повторить ошибки", self.retry),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            layout.addWidget(button)
        self.continue_errors = QCheckBox("Продолжать после ошибки")
        self.continue_errors.setChecked(True)
        layout.addWidget(self.continue_errors)
        self.status = QLabel(
            "Откройте каждый ролик, выберите его цель и добавьте в очередь. Настройки фиксируются отдельно для каждого задания. Очередь хранится в текущем сеансе."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        return group

    def add(self, job):
        if job.source.resolve() == job.destination.resolve():
            raise ValueError("Исходное видео нельзя перезаписывать")
        if any(e.job.destination.resolve() == job.destination.resolve() for e in self.entries):
            raise ValueError("В очереди уже есть такой файл результата")
        if any(e.job.source.resolve() == job.destination.resolve() for e in self.entries):
            raise ValueError("Результат заменит исходный ролик другого задания очереди")
        if any(e.job.destination.resolve() == job.source.resolve() for e in self.entries):
            raise ValueError("Этот исходный ролик будет перезаписан другим заданием очереди")
        job = replace(
            job,
            identity=job.identity or source_identity(job.source),
            frozen=True,
            queue_tag=job.queue_tag or uuid.uuid4().hex,
        )
        self.entries.append(QueueEntry(job))
        self.refresh()

    def add_current(self):
        main = self.main
        if main.active_source != "file" or main.video_path is None or main.latest_frame is None or main.busy:
            self.status.setText("Откройте ролик и дождитесь обработки выбранного кадра")
            return
        if not main.input_view.boxes or any(
            info.get("subject") is None for info in main.last_tracks.values()
        ):
            self.status.setText("Выберите цель в каждой рамке")
            return
        main.pause_source()
        try:
            self.studio.validate_analysis_source()
            info = self.studio.media.result
            if not info or self.studio.media.path != main.video_path:
                raise ValueError("Дождитесь проверки совместимости ролика")
            file, _ = QFileDialog.getSaveFileName(
                main,
                "Результат задания очереди",
                str(main.video_path.with_name(main.video_path.stem + "-framed.mp4")),
                "MP4 (*.mp4)",
            )
            if not file:
                return
            selections = main.current_selections()
            selection = selections if main.settings.slots else selections.get("single")
            settings = freeze_framing(main.settings, main.last_tracks)
            self.add(
                ExportJob(
                    main.video_path,
                    Path(file),
                    settings,
                    [30, 24, 15][main.fps_combo.currentIndex()],
                    main.last_render_time,
                    not main.mute.isChecked(),
                    selection,
                    identity=info["identity"],
                )
            )
            self.status.setText(f"Добавлено заданий: {len(self.entries)}")
        except (ValueError, OSError) as error:
            self.status.setText(str(error))

    def refresh(self):
        selected = self.list.currentItem().data(Qt.ItemDataRole.UserRole) if self.list.currentItem() else None
        self.list.clear()
        names = {
            "pending": "ожидает",
            "running": "обработка",
            "done": "готово",
            "error": "ошибка",
            "cancelled": "отменено",
        }
        for entry in self.entries:
            item = QListWidgetItem(f"{names[entry.state]} · {entry.job.source.name}")
            item.setData(Qt.ItemDataRole.UserRole, entry.job.queue_tag)
            item.setToolTip(str(entry.job.destination) + (f"\n{entry.error}" if entry.error else ""))
            self.list.addItem(item)
            if entry.job.queue_tag == selected:
                self.list.setCurrentItem(item)

    def start(self):
        if self.running:
            return
        if self.main.export_busy:
            self.status.setText("Дождитесь завершения текущего экспорта")
            return
        self.running = True
        self.next()

    def next(self):
        if not self.running or self.current is not None:
            return
        entry = next((e for e in self.entries if e.state == "pending"), None)
        if entry is None:
            self.running = False
            self.status.setText("Очередь завершена. Проверьте результаты и отмеченные ошибки.")
            return
        self.current = entry
        entry.state = "running"
        self.status.setText("Экспорт: " + entry.job.source.name)
        self.refresh()
        if not self.main.begin_export(entry.job):
            self.finished({"cancelled": True, "error": self.main.export_status.text()})

    def finished(self, result):
        if self.current is None:
            return
        entry, self.current = self.current, None
        entry.error = result.get("error", "")
        entry.state = "error" if entry.error else "cancelled" if result.get("cancelled") else "done"
        if entry.state == "cancelled" or (entry.state == "error" and not self.continue_errors.isChecked()):
            self.running = False
        self.refresh()
        if self.running:
            QTimer.singleShot(0, self.next)

    def stop(self):
        self.running = False
        if self.current is not None:
            self.main.export_worker.cancel.set()
        self.status.setText("Очередь остановлена; ожидающие задания сохранены в этом сеансе")

    def remove(self):
        selected = self.list.currentItem()
        if selected:
            tag = selected.data(Qt.ItemDataRole.UserRole)
            self.entries = [e for e in self.entries if e is self.current or e.job.queue_tag != tag]
            self.refresh()

    def retry(self):
        if self.running or self.current is not None:
            return
        for entry in self.entries:
            if entry.state in ("error", "cancelled"):
                entry.state, entry.error = "pending", ""
        self.refresh()
