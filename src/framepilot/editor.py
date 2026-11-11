"""Presentation-only editor shell. Existing controls keep their signals and state."""

import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .appearance import icon


class ElidedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.setFont(self.font())
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)


def label(text, name=None):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(
        name
        not in {
            "brandName",
            "brandCaption",
            "sectionEyebrow",
            "pageTitle",
            "inspectorTitle",
            "outputBadge",
            "previewTitle",
            "previewHint",
            "performanceText",
        }
    )
    return widget


def action(text, callback, symbol=None, primary=False):
    button = QPushButton(text)
    button.clicked.connect(callback)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        button.setProperty("primary", True)
    if symbol:
        button.setIcon(icon(symbol, "#123e33" if primary else "#a4bbcf"))
        button.setIconSize(QSize(17, 17))
    return button


def small_action(tooltip, callback, symbol):
    button = action("", callback, symbol)
    button.setObjectName("quietButton")
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip)
    button.setProperty("iconOnly", True)
    button.setFixedSize(32, 30)
    return button


class FoldSection(QFrame):
    def __init__(self, title, body, expanded=False):
        super().__init__()
        self.setObjectName("foldSection")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = QPushButton(
            "Калибровка приёмника" if title == "Калибровка принимающей программы" else title
        )
        self.header.setObjectName("foldHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.setAccessibleName(title)
        self.header.setToolTip(title)
        self.body = body
        body.setTitle("")
        body.setObjectName("sectionBody")
        body.layout().setContentsMargins(0, 0, 0, 0)
        body.layout().setSpacing(9)
        self.container = QWidget()
        content = QVBoxLayout(self.container)
        content.setContentsMargins(12, 0, 12, 12)
        content.addWidget(body)
        layout.addWidget(self.header)
        layout.addWidget(self.container)
        self.header.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.header.setChecked(expanded)
        self.header.setIcon(icon("down" if expanded else "chevron", "#89b6b2", 16))
        self.container.setVisible(expanded)


class PreviewCard(QFrame):
    def __init__(self, deck, role, title, detail, canvas):
        super().__init__()
        self.setObjectName("previewCard")
        self.role = role
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        header = QFrame()
        header.setObjectName("previewHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 5, 7, 5)
        marker = QLabel()
        marker.setPixmap(
            icon({"input": "source", "output": "output", "target": "frame"}[role], "#7db5ac", 16).pixmap(
                16, 16
            )
        )
        row.addWidget(marker)
        row.addWidget(label(title, "previewTitle"))
        row.addStretch()
        self.detail = label(detail, "previewHint")
        row.addWidget(self.detail)
        row.addWidget(small_action("Развернуть: " + title, lambda: deck.focus(role), "expand"))
        layout.addWidget(header)
        canvas.setProperty("previewRole", role)
        canvas.setProperty(
            "emptyDetail",
            {
                "output": "Добавьте видео и выберите цель для кадрирования",
                "input": "Видео, камера или перетащите файл в окно",
                "target": "Выберите окно на шаге «Рамка»",
            }[role],
        )
        layout.addWidget(canvas, 1)


class PreviewDeck(QWidget):
    def __init__(self, main):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(12)
        self.cards = {
            "output": PreviewCard(self, "output", "Результат", "", main.output_view),
            "input": PreviewCard(self, "input", "Исходник", "Выбор цели", main.input_view),
            "target": PreviewCard(self, "target", "Окно назначения", "Выбор рамки", main.target_view),
        }
        self.buttons = {}
        self.focused = None
        self.focus("output")

    def focus(self, role):
        if self.focused == role:
            return
        self.focused = role
        for card in self.cards.values():
            self.grid.removeWidget(card)
        self.grid.addWidget(self.cards[role], 0, 0, 1, 2)
        for column, key in enumerate(key for key in self.cards if key != role):
            self.grid.addWidget(self.cards[key], 1, column)
        self.grid.setRowStretch(0, 3)
        self.grid.setRowStretch(1, 2)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        for key, card in self.cards.items():
            card.setProperty("featured", key == role)
            card.style().unpolish(card)
            card.style().polish(card)
        if role in self.buttons:
            self.buttons[role].setChecked(True)


class EditorShell:
    PAGES = (
        ("Источник", "source", "Видео, фото и камера", "Добавьте файл или включите камеру."),
        ("Цель", "target", "Выбор и кадрирование", "Укажите цель на исходном изображении."),
        ("Рамка", "frame", "Место назначения", "Выберите окно и нужную рамку."),
        ("Вывод", "output", "Подключение и экспорт", "Настройте поток для принимающего приложения."),
    )

    def __init__(self, main, groups):
        self.main = main
        self.sections = {}
        self.root = QWidget()
        self.root.setObjectName("editorRoot")
        root = QVBoxLayout(self.root)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.build_header())
        body = QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(self.build_navigation())
        body.addWidget(self.build_workspace(), 1)
        body.addWidget(self.build_inspector(groups))
        root.addLayout(body, 1)
        root.addWidget(self.build_status())
        main.setCentralWidget(self.root)
        self.navigate(0, focus=False)
        main.player.playbackStateChanged.connect(self.refresh_transport)

    def build_header(self):
        main = self.main
        frame = QFrame()
        frame.setObjectName("topBar")
        row = QHBoxLayout(frame)
        row.setContentsMargins(20, 14, 20, 14)
        row.setSpacing(12)
        mark = QLabel()
        mark.setPixmap(icon("brand", "#78dfc5", 32).pixmap(32, 32))
        row.addWidget(mark)
        branding = QVBoxLayout()
        branding.setSpacing(1)
        branding.addWidget(label("FramePilot", "brandName"))
        branding.addWidget(label("Точный кадр. В реальном времени.", "brandCaption"))
        row.addLayout(branding)
        row.addStretch()
        self.output_badge = label("ВЫХОД ВЫКЛЮЧЕН", "outputBadge")
        self.output_badge.setFixedHeight(26)
        row.addWidget(self.output_badge)
        row.addSpacing(4)
        row.addWidget(action("Открыть файл", main.open_video, "folder", True))
        row.addWidget(action("В OBS", main.configure_obs, "output"))
        self.export_action = small_action("Экспортировать MP4", self.open_export, "export")
        row.addWidget(self.export_action)
        self.menu_button = QToolButton()
        self.menu_button.setObjectName("quietButton")
        self.menu_button.setIcon(icon("more"))
        self.menu_button.setToolTip("Профили и помощь")
        self.menu_button.setAccessibleName("Профили и помощь")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.menu_button)
        menu.addAction("Мастер подключения…", main.open_connection_wizard)
        camera_name = menu.addAction("Имя виртуальной камеры (Windows)…", main.configure_camera_name)
        camera_name.setVisible(sys.platform == "win32")
        menu.addAction("Сохранить пример ошибки (кадры + параметры)…", main.save_debug_case)
        menu.addAction("Сохранить профиль…", main.save_profile)
        menu.addAction("Открыть профиль…", main.load_profile)
        menu.addSeparator()
        menu.addAction("Горячие клавиши", main.show_shortcuts)
        self.menu_button.setMenu(menu)
        row.addWidget(self.menu_button)
        main.black_button = action("Чёрный кадр", main.toggle_black_output, "black")
        main.black_button.setObjectName("blackOutput")
        main.black_button.setToolTip("Закрыть / вернуть видеовыход · Cmd+Shift+X")
        row.addWidget(main.black_button)
        return frame

    def build_navigation(self):
        rail = QFrame()
        rail.setObjectName("navigationRail")
        rail.setFixedWidth(164)
        column = QVBoxLayout(rail)
        column.setContentsMargins(12, 23, 12, 17)
        column.setSpacing(7)
        column.addWidget(label("РАБОЧИЙ ПРОЦЕСС", "sectionEyebrow"))
        column.addSpacing(9)
        self.navigation = QButtonGroup(rail)
        self.nav_buttons = []
        for index, (title, symbol, _, _) in enumerate(self.PAGES):
            button = action(title, lambda checked=False, i=index: self.navigate(i), symbol)
            button.setObjectName("navStep")
            button.setCheckable(True)
            button.setMinimumHeight(44)
            self.navigation.addButton(button, index)
            self.nav_buttons.append(button)
            column.addWidget(button)
        column.addStretch()
        scenarios = action("Сценарии", lambda: self.open_section(0, "Сценарии"), "layers")
        scenarios.setObjectName("quietButton")
        column.addWidget(scenarios)
        examples = QToolButton()
        examples.setText("Примеры")
        examples.setIcon(icon("source"))
        examples.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        examples.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        examples.setObjectName("quietButton")
        menu = QMenu(examples)
        menu.addAction("Лицо · тестовый портрет", self.main.load_sample)
        menu.addAction("Предмет · синтетический ролик", self.main.load_object_sample)
        menu.addAction("Тестовое принимающее окно", self.main.show_demo)
        examples.setMenu(menu)
        column.addWidget(examples)
        column.addSpacing(14)
        column.addWidget(label("Локальная обработка", "mutedText"))
        column.addWidget(label("FramePilot  " + __version__, "brandCaption"))
        return rail

    def build_workspace(self):
        main = self.main
        workspace = QWidget()
        column = QVBoxLayout(workspace)
        column.setContentsMargins(18, 20, 18, 16)
        column.setSpacing(13)
        heading = QHBoxLayout()
        caption = QVBoxLayout()
        caption.setSpacing(3)
        caption.addWidget(label("Ваша видеостудия", "pageTitle"))
        hint = label("Источник → выбранная цель → готовый кадр", "mutedText")
        hint.setWordWrap(False)
        caption.addWidget(hint)
        heading.addLayout(caption)
        heading.addStretch()
        switch = QFrame()
        switch.setObjectName("modeSwitch")
        switch_row = QHBoxLayout(switch)
        switch_row.setContentsMargins(3, 3, 3, 3)
        switch_row.setSpacing(2)
        self.deck = PreviewDeck(main)
        self.preview_modes = QButtonGroup(switch)
        for role, text in (("output", "Выход"), ("input", "Цель"), ("target", "Рамка")):
            button = action(text, lambda checked=False, r=role: self.deck.focus(r))
            button.setObjectName("previewMode")
            button.setCheckable(True)
            self.preview_modes.addButton(button)
            self.deck.buttons[role] = button
            switch_row.addWidget(button)
        self.deck.buttons["output"].setChecked(True)
        heading.addWidget(switch)
        column.addLayout(heading)
        column.addWidget(self.deck, 1)
        transport = QFrame()
        transport.setObjectName("transport")
        controls = QVBoxLayout(transport)
        controls.setContentsMargins(12, 9, 12, 9)
        controls.setSpacing(5)
        source_row = QHBoxLayout()
        source_row.addWidget(main.filename, 1)
        self.output_summary = label("1280 × 720 · 30 FPS", "previewHint")
        source_row.addWidget(self.output_summary)
        controls.addLayout(source_row)
        timeline = QHBoxLayout()
        timeline.setSpacing(10)
        main.play_button = action("", main.toggle_play, "play", True)
        main.play_button.setFixedSize(36, 33)
        main.play_button.setAccessibleName("Воспроизвести / пауза")
        timeline.addWidget(main.play_button)
        self.restart_action = small_action("С начала", main.restart, "restart")
        timeline.addWidget(self.restart_action)
        timeline.addWidget(main.timeline, 1)
        main.position_label.setObjectName("performanceText")
        timeline.addWidget(main.position_label)
        timeline.addWidget(main.loop)
        timeline.addWidget(main.mute)
        controls.addLayout(timeline)
        controls.addWidget(main.studio.markers)
        column.addWidget(transport)
        return workspace

    def source_controls(self):
        from PySide6.QtWidgets import QGroupBox

        main = self.main
        group = QGroupBox()
        column = QVBoxLayout(group)
        column.setSpacing(10)
        column.addWidget(action("Выбрать видеофайл…", main.open_video, "folder", True))
        column.addWidget(action("Открыть фото…", main.open_photo, "folder"))
        column.addWidget(label("Или перетащите видео/фото в рабочую область.", "mutedText"))
        column.addSpacing(12)
        column.addWidget(label("ВХОД С КАМЕРЫ", "sectionEyebrow"))
        row = QHBoxLayout()
        row.addWidget(main.camera_combo, 1)
        row.addWidget(small_action("Обновить список камер", main.refresh_cameras, "refresh"))
        column.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(action("Включить", main.start_camera, "camera"))
        row.addWidget(action("Выключить", main.stop_camera))
        column.addLayout(row)
        column.addWidget(label("Камера включается только по вашему действию.", "mutedText"))
        return group

    def build_inspector(self, groups):
        frame = QFrame()
        frame.setObjectName("inspector")
        frame.setFixedWidth(338)
        column = QVBoxLayout(frame)
        column.setContentsMargins(0, 0, 0, 0)
        self.inspector_pages = QStackedWidget()
        self.page_layouts = []
        self.scrolls = []
        for _, _, title, hint in self.PAGES:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(16, 22, 16, 18)
            layout.setSpacing(12)
            layout.addWidget(label(title, "inspectorTitle"))
            layout.addWidget(label(hint, "mutedText"))
            scroll.setWidget(page)
            self.inspector_pages.addWidget(scroll)
            self.page_layouts.append(layout)
            self.scrolls.append(scroll)
        self.add_section(0, "Ваш источник", self.source_controls(), True)
        routes = {
            "Фото": (1, False),
            "Совместимость видео": (0, False),
            "Восстановление сеанса": (0, False),
            "Очередь экспорта": (3, False),
            "Захват экрана и окна": (0, False),
            "Анализ ролика": (1, False),
            "Качество и поворот": (1, False),
            "Поправки по времени": (1, False),
            "Маска и форма рамки": (2, False),
            "Запись выходного потока": (3, False),
            "Сценарии": (0, False),
            "Файлы профилей": (0, False),
            "Цель и отслеживание": (1, True),
            "Размещение": (1, True),
            "Автоматическая смена плана": (1, False),
            "Движение и ориентиры": (1, False),
            "Фон композиции": (1, False),
            "Рамка в приложении": (2, True),
            "Рамки композиции · до 4": (2, False),
            "Профили окон": (2, False),
            "Калибровка принимающей программы": (2, False),
            "Вывод в OBS": (3, True),
            "Если пропал видеосигнал": (3, False),
            "Экспорт видео": (3, False),
        }
        order = [
            "Фото",
            "Совместимость видео",
            "Восстановление сеанса",
            "Захват экрана и окна",
            "Сценарии",
            "Файлы профилей",
            "Цель и отслеживание",
            "Размещение",
            "Качество и поворот",
            "Анализ ролика",
            "Поправки по времени",
            "Автоматическая смена плана",
            "Движение и ориентиры",
            "Фон композиции",
            "Рамка в приложении",
            "Маска и форма рамки",
            "Рамки композиции · до 4",
            "Профили окон",
            "Калибровка принимающей программы",
            "Вывод в OBS",
            "Экспорт видео",
            "Очередь экспорта",
            "Запись выходного потока",
            "Если пропал видеосигнал",
        ]
        by_title = {group.title(): group for group in groups}
        for title in order:
            if title in by_title:
                index, expanded = routes[title]
                self.add_section(index, title, by_title.pop(title), expanded)
        # Any future group remains reachable even before a dedicated routing choice.
        for title, group in by_title.items():
            self.add_section(3, title, group, False)
        for layout in self.page_layouts:
            layout.addStretch()
        column.addWidget(self.inspector_pages)
        return frame

    def add_section(self, index, title, body, expanded):
        section = FoldSection(title, body, expanded)
        self.sections[title] = section
        self.page_layouts[index].addWidget(section)

    def build_status(self):
        frame = QFrame()
        frame.setObjectName("statusBar")
        row = QHBoxLayout(frame)
        row.setContentsMargins(20, 9, 20, 9)
        row.setSpacing(20)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(self.main.signal_label)
        text.addWidget(self.main.status)
        row.addLayout(text, 1)
        detail = QVBoxLayout()
        detail.setSpacing(2)
        detail.addWidget(self.main.quality_label)
        detail.addWidget(self.main.metrics)
        row.addLayout(detail, 1)
        return frame

    def navigate(self, index, focus=True):
        self.inspector_pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        if focus:
            self.deck.focus({0: "output", 1: "input", 2: "target", 3: "output"}[index])

    def open_section(self, index, title):
        self.navigate(index)
        section = self.sections[title]
        section.set_expanded(True)
        self.scrolls[index].ensureWidgetVisible(section)

    def open_export(self):
        if self.main.active_source == "photo":
            self.open_section(1, "Фото")
            self.main.save_photo_dialog()
            return
        self.open_section(3, "Экспорт видео")
        self.main.start_export()

    def refresh_transport(self, *_):
        from PySide6.QtMultimedia import QMediaPlayer

        main = self.main
        photo = main.active_source == "photo"
        main.play_button.setEnabled(not photo)
        self.restart_action.setEnabled(not photo)
        main.loop.setEnabled(main.active_source == "file")
        main.mute.setEnabled(main.active_source == "file")
        self.export_action.setToolTip("Сохранить PNG / JPEG" if photo else "Экспортировать MP4")
        playing = (
            main.studio.screen.running and not main.studio.screen.paused
            if main.active_source == "screen"
            else main.camera_input.camera is not None and not main.camera_input.paused
            if main.active_source == "camera"
            else main.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        if getattr(self, "playing", None) != playing:
            self.playing = playing
            main.play_button.setIcon(icon("pause" if playing else "play", "#123e33"))
            main.play_button.setToolTip("Пауза · пробел" if playing else "Воспроизвести · пробел")

    def refresh_status(self):
        main = self.main
        self.refresh_transport()
        self.sections["Фото"].setVisible(main.active_source == "photo")
        size = f"{main.settings.width} × {main.settings.height}"
        fps = [30, 24, 15][main.fps_combo.currentIndex()]
        self.output_summary.setText(f"{size} · {fps} FPS")
        warning = bool(main.last_composition.get("warnings"))
        self.deck.cards["output"].detail.setText(size + (" · !" if warning else ""))
        self.deck.cards["output"].detail.setToolTip("\n".join(main.last_composition.get("warnings", [])))
        active = main.publisher.url is not None
        if self.output_badge.property("active") != active:
            self.output_badge.setProperty("active", active)
            self.output_badge.style().unpolish(self.output_badge)
            self.output_badge.style().polish(self.output_badge)
        self.output_badge.setText("ЛОКАЛЬНЫЙ ВЫХОД" if active else "ВЫХОД ВЫКЛЮЧЕН")
        if main.black_button.property("active") != main.emergency_black:
            main.black_button.setProperty("active", main.emergency_black)
            main.black_button.style().unpolish(main.black_button)
            main.black_button.style().polish(main.black_button)
