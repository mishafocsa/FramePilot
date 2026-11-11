import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QPushButton

from framepilot.app import MainWindow
from framepilot.appearance import install_appearance
from framepilot.geometry import Rect
from framepilot.widgets import to_image


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    install_appearance(app)
    return app


@pytest.fixture
def window(application):
    window = MainWindow()
    window.show()
    application.processEvents()
    try:
        yield window
    finally:
        window.close()
        application.processEvents()


def settle(application, predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_navigation_keeps_real_selected_object_and_canvas_coordinates(application, window):
    image = np.zeros((180, 320, 3), np.uint8)
    image[40:120, 50:170] = np.random.default_rng(9).integers(20, 255, (80, 120, 3), np.uint8)
    subject = Rect(50 / 320, 40 / 180, 120 / 320, 80 / 180)
    window.latest_frame = image
    window.input_view.show_image(to_image(image))
    window.select_object(subject)
    settle(application, lambda: not window.busy and bool(window.input_view.boxes))
    window.subject_region_combo.setCurrentIndex(1)
    settle(application, lambda: not window.busy and bool(window.input_view.focus))
    tracked = window.video_worker.pipeline.object_tracker.box
    seed = window.object_seed
    settings = window.settings
    for page, role in ((1, "input"), (2, "target"), (3, "output"), (0, "output"), (1, "input")):
        QTest.mouseClick(window.editor.nav_buttons[page], Qt.MouseButton.LeftButton)
        application.processEvents()
        assert window.editor.inspector_pages.currentIndex() == page
        assert window.editor.deck.focused == role
        assert all(card.isVisible() for card in window.editor.deck.cards.values())
        center = window.input_view.area().center()
        assert window.input_view.normalized(QPointF(center)) == pytest.approx((0.5, 0.5))
        assert window.video_worker.pipeline.object_tracker.box == tracked
        assert window.object_seed == seed
        assert window.settings == settings


@pytest.mark.parametrize("size", [(1120, 740), (1380, 900)])
def test_every_inspector_page_fits_without_horizontal_clipping(application, window, size):
    window.resize(*size)
    for section in window.editor.sections.values():
        section.set_expanded(True)
    for page in range(4):
        window.editor.navigate(page)
        application.processEvents()
        scroll = window.editor.scrolls[page]
        assert scroll.widget().width() <= scroll.viewport().width()
        for kind in (QPushButton, QComboBox, QCheckBox, QDoubleSpinBox):
            for control in scroll.widget().findChildren(kind):
                if not control.isVisible():
                    continue
                scroll.ensureWidgetVisible(control)
                application.processEvents()
                origin = control.mapTo(scroll.viewport(), QPoint(0, 0))
                assert origin.x() >= 0
                assert origin.x() + control.width() <= scroll.viewport().width()
                assert control.width() >= control.minimumSizeHint().width()
        assert window.black_button.isVisible()
        assert window.black_button.mapTo(window, QPoint(0, 0)).y() < 100


def test_collapsed_settings_keep_values_and_export_routes_to_actual_controls(
    application, window, monkeypatch
):
    window.subject_region_combo.setCurrentIndex(1)
    window.editor.open_section(1, "Автоматическая смена плана")
    window.auto_framing_enabled.setChecked(True)
    window.frame_threshold.setValue(35)
    window.padding.setValue(0.4)
    section = window.editor.sections["Автоматическая смена плана"]
    QTest.mouseClick(section.header, Qt.MouseButton.LeftButton)
    window.editor.navigate(3)
    window.editor.open_section(1, "Автоматическая смена плана")
    application.processEvents()
    assert section.container.isVisible()
    assert window.frame_threshold.value() == 35
    assert window.settings.framing.enabled
    assert window.settings.framing.threshold == pytest.approx(0.35)
    assert window.settings.padding == pytest.approx(0.4)
    calls = []
    monkeypatch.setattr(window, "start_export", lambda: calls.append(window.export_button.isVisible()))
    window.editor.open_export()
    assert window.editor.inspector_pages.currentIndex() == 3
    assert window.editor.sections["Экспорт видео"].container.isVisible()
    assert calls == [True]


def test_persistent_black_button_closes_and_restores_real_preview(application, window):
    window.preferred_port = 0
    window.start_local_output()
    window.editor.refresh_status()
    assert window.editor.output_badge.property("active")
    assert "ЛОКАЛЬНЫЙ" in window.editor.output_badge.text()
    for page in (0, 1, 2, 3):
        window.editor.navigate(page)
        QTest.mouseClick(window.black_button, Qt.MouseButton.LeftButton)
        window.watch_signal()
        assert window.emergency_black
        assert window.publisher.hold_reason == "emergency_black"
        assert window.black_button.property("active")
        assert "Вернуть" in window.black_button.text()
        QTest.mouseClick(window.black_button, Qt.MouseButton.LeftButton)
        window.watch_signal()
        assert not window.emergency_black
        assert not window.black_button.property("active")
    window.stop_local_output()
    window.editor.refresh_status()
    assert not window.editor.output_badge.property("active")
