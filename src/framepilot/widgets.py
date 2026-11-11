import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .appearance import paint_empty_preview
from .calibration import Calibration, calibration_card, received_preview
from .geometry import Rect


def to_image(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def to_array(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    h, w, stride = image.height(), image.width(), image.bytesPerLine()
    rgb = np.frombuffer(image.constBits(), dtype=np.uint8, count=h * stride).reshape(h, stride)
    rgb = rgb[:, : w * 3].reshape(h, w, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class ImageCanvas(QWidget):
    clicked = Signal(object)
    selected = Signal(object)

    def __init__(self, placeholder: str = "", drag: bool = False):
        super().__init__()
        self.image = QImage()
        self.boxes: list[Rect] = []
        self.candidates: list[Rect] = []
        self.markers = []
        self.recovery = []
        self.focus = []
        self.polygons = []
        self.placeholder = placeholder
        self.drag = drag
        self.start: QPointF | None = None
        self.end: QPointF | None = None
        self.setMinimumSize(240, 150)
        self.setMouseTracking(True)

    def show_image(self, image: QImage):
        self.image = image
        self.update()

    def area(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def normalized(self, p: QPointF) -> tuple[float, float] | None:
        area = self.area()
        if area.isEmpty() or not area.contains(p):
            return None
        return (p.x() - area.x()) / area.width(), (p.y() - area.y()) / area.height()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080b11"))
        if self.image.isNull():
            if self.property("previewRole"):
                paint_empty_preview(
                    painter,
                    self.rect(),
                    self.property("previewRole"),
                    self.placeholder,
                    self.property("emptyDetail") or "",
                )
                return
            painter.setPen(QColor("#8896ac"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        area = self.area()
        painter.drawImage(area, self.image)
        painter.setPen(QPen(QColor("#7890aa"), 1, Qt.PenStyle.DashLine))
        for index, r in enumerate(self.candidates, 1):
            box = QRectF(
                area.x() + r.x * area.width(),
                area.y() + r.y * area.height(),
                r.w * area.width(),
                r.h * area.height(),
            )
            painter.drawRect(box)
            painter.drawText(box.topLeft() + QPointF(5, 17), str(index))
        painter.setPen(QPen(QColor("#51e0b4"), 2))
        for r in self.boxes:
            painter.drawRect(
                QRectF(
                    area.x() + r.x * area.width(),
                    area.y() + r.y * area.height(),
                    r.w * area.width(),
                    r.h * area.height(),
                )
            )
        for x, y in self.markers:
            point = QPointF(area.x() + x * area.width(), area.y() + y * area.height())
            painter.drawLine(point + QPointF(-12, 0), point + QPointF(12, 0))
        painter.setPen(QPen(QColor("#69b9ff"), 2, Qt.PenStyle.DashLine))
        for rect in self.focus:
            painter.drawRect(
                QRectF(
                    area.x() + rect.x * area.width(),
                    area.y() + rect.y * area.height(),
                    rect.w * area.width(),
                    rect.h * area.height(),
                )
            )
        painter.setPen(QPen(QColor("#ffd166"), 2, Qt.PenStyle.DashLine))
        for polygon in self.polygons:
            painter.drawPolygon(
                QPolygonF(
                    [QPointF(area.x() + x * area.width(), area.y() + y * area.height()) for x, y in polygon]
                )
            )
        for index, rect in enumerate(self.recovery, 1):
            box = QRectF(
                area.x() + rect.x * area.width(),
                area.y() + rect.y * area.height(),
                rect.w * area.width(),
                rect.h * area.height(),
            )
            painter.drawRect(box)
            painter.drawText(box.topLeft() + QPointF(5, 17), f"?{index}")
        if self.start is not None and self.end is not None:
            painter.setPen(QPen(QColor("#ffd166"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self.start, self.end).normalized().intersected(area))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.normalized(event.position()) is None:
            return
        self.start = self.end = event.position()
        if not self.drag:
            self.clicked.emit(self.normalized(event.position()))

    def mouseMoveEvent(self, event):
        if self.drag and self.start is not None:
            self.end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag and self.start is not None:
            rect = QRectF(self.start, event.position()).normalized().intersected(self.area())
            area = self.area()
            if not area.isEmpty() and rect.width() > 10 and rect.height() > 10:
                self.selected.emit(
                    Rect(
                        (rect.x() - area.x()) / area.width(),
                        (rect.y() - area.y()) / area.height(),
                        rect.width() / area.width(),
                        rect.height() / area.height(),
                    )
                )
            elif (event.position() - self.start).manhattanLength() < 8:
                point = self.normalized(event.position())
                if point is not None:
                    self.clicked.emit(point)
        self.start = self.end = None
        self.update()


class DemoTarget(QWidget):
    """An actual receiving surface with changing slots, used by the normal contour detector."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FramePilot · тестовое приложение")
        self.resize(800, 500)
        self.setMinimumSize(400, 300)
        self.image = QImage()
        self.mapping = "camera"
        self.calibration = Calibration()
        self.index = 0
        self.animate = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.timer.start(8000)

    def advance(self):
        if self.animate:
            self.index = (self.index + 1) % 4
            self.update()

    def slot(self) -> tuple[Rect, str]:
        return [
            (Rect(0.32, 0.12, 0.36, 0.76), "rectangle"),
            (Rect(0.08, 0.25, 0.6, 0.5), "rectangle"),
            (Rect(0.6, 0.12, 0.28, 0.7), "rectangle"),
            (Rect(0.35, 0.13, 0.3, 0.72), "ellipse"),
        ][self.index]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#222936"))
        slot, shape = self.slot()
        r = slot.pixels(self.width(), self.height())
        area = QRectF(r.x, r.y, r.w, r.h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#07090f"))
        if shape == "ellipse":
            painter.drawEllipse(area)
        else:
            painter.drawRect(area)
        if not self.image.isNull():
            picture = (
                to_image(received_preview(to_array(self.image), self.calibration))
                if self.mapping == "camera"
                else self.image
            )
            painter.save()
            if shape == "ellipse":
                from PySide6.QtGui import QPainterPath

                clip = QPainterPath()
                clip.addEllipse(area)
                painter.setClipPath(clip)
            else:
                painter.setClipRect(area)
            destination = area if self.mapping == "camera" else QRectF(self.rect())
            scale = min(destination.width() / picture.width(), destination.height() / picture.height())
            iw, ih = picture.width() * scale, picture.height() * scale
            fitted = QRectF(destination.center().x() - iw / 2, destination.center().y() - ih / 2, iw, ih)
            painter.drawImage(fitted, picture)
            painter.restore()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#ffffff"), 5))
        if shape == "ellipse":
            painter.drawEllipse(area)
        else:
            painter.drawRect(area)

    def closeEvent(self, event):
        # Reusable target; do not destroy its timer or receiving surface.
        self.hide()
        event.ignore()


class OBSSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Подключение OBS")
        layout = QVBoxLayout(self)
        help_ = QLabel(
            "В OBS откройте «Сервис → Настройки сервера WebSocket», включите сервер "
            "и посмотрите порт и пароль. Подключение выполняется к этому компьютеру.\n\n"
            "Будут созданы отдельная сцена FramePilot и источник FramePilot Video. "
            "После подключения выберите сцену FramePilot в OBS. Для передачи в другое "
            "приложение включите виртуальную камеру флажком ниже. "
            "FramePilot должен оставаться запущенным."
        )
        help_.setWordWrap(True)
        layout.addWidget(help_)
        form = QFormLayout()
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(4455)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Порт OBS WebSocket", self.port)
        form.addRow("Пароль (не сохраняется)", self.password)
        layout.addLayout(form)
        self.activate = QCheckBox("Переключить OBS на FramePilot и включить виртуальную камеру")
        layout.addWidget(self.activate)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Передать в OBS")
        layout.addWidget(buttons)
        self.resize(540, 280)


class CalibrationDialog(QDialog):
    def __init__(self, current: Calibration, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Калибровка принимающей программы")
        layout = QVBoxLayout(self)
        description = QLabel(
            "Укажите, какие края изображения камеры обрезает принимающая программа. "
            "Укажите её зеркалирование и поворот. Сетка помогает сравнить края и надписи LEFT/RIGHT. "
            "Сетка моделирует обрезку, зеркало и поворот; масштаб и смещение оценивайте на видео. "
            "Это модель заданных параметров, а не измерение чужой программы."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.edges = []
        for label, value in [
            ("Слева, %", current.crop.x * 100),
            ("Сверху, %", current.crop.y * 100),
            ("Справа, %", (1 - current.crop.x - current.crop.w) * 100),
            ("Снизу, %", (1 - current.crop.y - current.crop.h) * 100),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(0, 45)
            spin.setValue(value)
            spin.valueChanged.connect(self.refresh)
            self.edges.append(spin)
            form.addRow(label, spin)
        self.mirror = QCheckBox("Программа зеркалит изображение — компенсировать")
        self.mirror.setChecked(current.mirror)
        self.mirror.toggled.connect(self.refresh)
        form.addRow(self.mirror)
        self.rotation = QComboBox()
        for angle in (0, 90, 180, 270):
            self.rotation.addItem(f"{angle}° по часовой стрелке", angle)
        self.rotation.setCurrentIndex(self.rotation.findData(current.rotation))
        self.rotation.currentIndexChanged.connect(self.refresh)
        form.addRow("Приёмник поворачивает", self.rotation)
        self.adjustments = []
        for label, value, low, high in [
            ("Дополнительный масштаб", current.zoom, 0.5, 2),
            ("Смещение по X (доля рамки)", current.offset_x, -0.5, 0.5),
            ("Смещение по Y (доля рамки)", current.offset_y, -0.5, 0.5),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(0.05)
            spin.setValue(value)
            self.adjustments.append(spin)
            form.addRow(label, spin)
        layout.addLayout(form)
        self.preview = ImageCanvas()
        self.preview.setFixedHeight(180)
        layout.addWidget(self.preview)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def value(self):
        left, top, right, bottom = [spin.value() / 100 for spin in self.edges]
        return Calibration(
            Rect(left, top, 1 - left - right, 1 - top - bottom),
            self.mirror.isChecked(),
            *[spin.value() for spin in self.adjustments],
            self.rotation.currentData(),
        )

    def refresh(self, *_):
        if hasattr(self, "preview"):
            self.preview.show_image(to_image(received_preview(calibration_card(640, 360), self.value())))


class ExportDialog(QDialog):
    def __init__(self, object_mode=False, parent=None, scene_mode=False):
        super().__init__(parent)
        self.setWindowTitle("Экспорт обработанного MP4")
        layout = QVBoxLayout(self)
        label = QLabel(
            "Будет сохранён обработанный видеовыход с текущей калибровкой. "
            "Экспорт не зависит от скорости воспроизведения и не пропускает кадры. "
            "Настройки фиксируются на момент запуска экспорта."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.start = QComboBox()
        self.start.addItems(
            ["С текущего кадра — все рамки"]
            if scene_mode
            else ["С кадра выбора предмета"]
            if object_mode
            else ["С начала — самое крупное лицо", "С текущего — выбранное лицо"]
        )
        layout.addWidget(self.start)
        self.audio = QCheckBox("Сохранить исходный звук, если он есть")
        self.audio.setChecked(True)
        layout.addWidget(self.audio)
        self.fps = QComboBox()
        self.fps.addItems(["30 FPS", "24 FPS", "15 FPS"])
        layout.addWidget(self.fps)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
