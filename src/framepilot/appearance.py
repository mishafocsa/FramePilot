"""Shared native Qt appearance and small, code-drawn icons."""

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap

STYLE = """
QWidget { color: #e6edf4; font-size: 13px; }
QMainWindow, QDialog { background: #0c1118; }
QWidget#editorRoot { background: #0c1118; }
QFrame#topBar { background: #111821; border-bottom: 1px solid #222d39; }
QFrame#navigationRail { background: #101720; border-right: 1px solid #222d39; }
QFrame#inspector { background: #111821; border-left: 1px solid #222d39; }
QFrame#statusBar { background: #101720; border-top: 1px solid #222d39; }
QLabel { background: transparent; }
QLabel#brandName { font-size: 18px; font-weight: 650; letter-spacing: -0.4px; color: #f3f7fb; }
QLabel#brandCaption { color: #8a9aab; font-size: 11px; }
QLabel#sectionEyebrow { color: #7c90a5; font-size: 10px; font-weight: 600; letter-spacing: 1.1px; }
QLabel#pageTitle { font-size: 22px; font-weight: 600; letter-spacing: -0.5px; color: #f4f7fa; }
QLabel#inspectorTitle { font-size: 19px; font-weight: 600; color: #f4f7fa; }
QLabel#mutedText { color: #8ea0b3; font-size: 12px; }
QLabel#performanceText { color: #91a5bb; font-size: 11px; }
QLabel#outputBadge { color: #a5b3c2; background: #1a2531; border: 1px solid #2b3b4c; border-radius: 10px; padding: 4px 9px; font-size: 10px; font-weight: 600; }
QLabel#outputBadge[active=true] { color: #91eed2; background: #14332e; border-color: #235147; }
QPushButton, QToolButton { background: #1b2734; color: #dbe5ee; border: 1px solid #304153; border-radius: 8px; padding: 8px 12px; font-weight: 500; }
QPushButton:hover, QToolButton:hover { background: #243447; border-color: #4d647b; }
QPushButton:pressed, QToolButton:pressed { background: #13202c; }
QPushButton:focus, QToolButton:focus { border-color: #60d8ba; }
QPushButton:disabled, QToolButton:disabled { color: #6e8093; background: #151f2a; border-color: #253241; }
QPushButton[primary=true] { color: #092a24; background: #70e1c2; border-color: #70e1c2; font-weight: 600; }
QPushButton[primary=true]:hover { background: #91edd4; border-color: #91edd4; }
QPushButton[primary=true]:disabled { background: #26483f; color: #80a99b; border-color: #33564c; }
QPushButton#blackOutput { background: #281c26; color: #edb3bb; border-color: #573241; }
QPushButton#blackOutput:hover { background: #3b2330; border-color: #a15d70; }
QPushButton#blackOutput[active=true] { background: #c64c66; color: #ffffff; border-color: #e27085; }
QPushButton#navStep { background: transparent; border: 1px solid transparent; text-align: left; padding: 12px 13px; color: #91a4b8; border-radius: 9px; }
QPushButton#navStep:hover { background: #172432; color: #e6edf4; }
QPushButton#navStep:checked { background: #15362f; color: #9af0d8; border-color: #245348; }
QPushButton#quietButton, QToolButton#quietButton { background: transparent; border-color: transparent; color: #97abc0; }
QPushButton#quietButton:hover, QToolButton#quietButton:hover { background: #1e2b39; color: #e6edf4; }
QPushButton[iconOnly=true] { padding: 4px; }
QFrame#inspector QPushButton { padding: 7px 8px; }
QFrame#inspector QPushButton[iconOnly=true] { padding: 4px; }
QPushButton#previewMode { background: transparent; border-color: transparent; border-radius: 6px; padding: 6px 10px; color: #8fa3b9; }
QPushButton#previewMode:checked { background: #2a3b4d; color: #f4f7fb; border-color: #3c5168; }
QFrame#modeSwitch { background: #151f2b; border: 1px solid #273647; border-radius: 9px; }
QFrame#previewCard { background: #141d28; border: 1px solid #293748; border-radius: 12px; }
QFrame#previewCard[featured=true] { border-color: #3b5d67; }
QFrame#previewHeader { background: transparent; border: none; }
QLabel#previewTitle { font-size: 12px; font-weight: 600; color: #e0e9f2; }
QLabel#previewHint { font-size: 10px; color: #859bb2; }
QFrame#transport { background: #141e29; border: 1px solid #2a394b; border-radius: 11px; }
QGroupBox { background: #151f2b; border: 1px solid #29394a; border-radius: 11px; margin-top: 11px; padding: 16px 12px 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 4px; color: #cbd9e6; font-weight: 600; }
QGroupBox#sectionBody { background: transparent; border: none; padding: 2px 0 0; margin: 0; }
QFrame#foldSection { background: #151f2b; border: 1px solid #29394a; border-radius: 11px; }
QPushButton#foldHeader { text-align: left; background: transparent; border: none; padding: 12px; font-weight: 600; color: #cbd9e6; }
QPushButton#foldHeader:hover { color: #a0efd9; }
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QPlainTextEdit { background: #101925; border: 1px solid #304356; border-radius: 7px; padding: 7px 9px; color: #e1eaf3; selection-background-color: #245d52; }
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QPlainTextEdit:focus { border-color: #63cdb0; }
QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled { color: #728399; border-color: #253342; }
QComboBox::drop-down { border: none; width: 23px; }
QComboBox::down-arrow { image: url(@ASSETS@/chevron-down.svg); width: 12px; height: 12px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { border: none; width: 22px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { border: none; width: 22px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@ASSETS@/chevron-up.svg); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@ASSETS@/chevron-down.svg); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: #1b2837; border: 1px solid #3b5065; selection-background-color: #255c50; outline: none; padding: 4px; }
QCheckBox { spacing: 8px; color: #b6c7d8; background: transparent; padding: 4px 0; }
QCheckBox:disabled { color: #71869b; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 4px; border: 1px solid #486077; background: #111c28; }
QCheckBox::indicator:checked { background: #63cfb3; border-color: #80e1c7; image: url(@ASSETS@/check.svg); }
QCheckBox::indicator:disabled { border-color: #2c3c4e; background: #17232f; }
QSlider { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: #2a3a4d; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #64d6b7; border-radius: 2px; }
QSlider::handle:horizontal { background: #d3f5eb; border: 2px solid #63cdb0; width: 10px; margin: -5px 0; border-radius: 7px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 3px 0; }
QScrollBar::handle:vertical { background: #364b62; min-height: 28px; border-radius: 3px; }
QScrollBar::handle:vertical:hover { background: #4d6985; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QSplitter::handle { background: #1e2c3b; width: 3px; }
QTabWidget::pane { background: #111a25; border: 1px solid #2b3b4d; border-radius: 8px; padding: 4px; }
QTabBar::tab { background: transparent; color: #91a6bc; padding: 10px 12px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #8ee9cf; border-bottom-color: #70e1c2; }
QTabBar::tab:hover { color: #e3edf6; background: #1b2938; }
QMenu { background: #182432; color: #deebf5; border: 1px solid #364a60; padding: 6px; border-radius: 8px; }
QMenu::item { padding: 8px 22px 8px 12px; border-radius: 5px; }
QMenu::item:selected { background: #274e48; }
QMenu::separator { height: 1px; background: #324355; margin: 5px; }
QToolButton::menu-indicator { image: none; width: 0; height: 0; }
QToolTip { background: #25374a; color: #edf4fa; border: 1px solid #4b627b; padding: 6px; }
QProgressBar { background: #111d29; border: 1px solid #304255; border-radius: 6px; min-height: 14px; text-align: center; }
QProgressBar::chunk { background: #397d6e; border-radius: 5px; }
""".replace("@ASSETS@", (Path(__file__).resolve().parent / "assets").as_posix())


def install_appearance(application):
    application.setStyle("Fusion")
    application.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
    palette = QPalette()
    for role, value in (
        (QPalette.ColorRole.Window, "#0c1118"),
        (QPalette.ColorRole.WindowText, "#e6edf4"),
        (QPalette.ColorRole.Base, "#101925"),
        (QPalette.ColorRole.AlternateBase, "#192635"),
        (QPalette.ColorRole.Text, "#e6edf4"),
        (QPalette.ColorRole.Button, "#1b2734"),
        (QPalette.ColorRole.ButtonText, "#e6edf4"),
        (QPalette.ColorRole.Highlight, "#2d6d5f"),
        (QPalette.ColorRole.HighlightedText, "#f1faf7"),
        (QPalette.ColorRole.ToolTipBase, "#25374a"),
        (QPalette.ColorRole.ToolTipText, "#edf4fa"),
        (QPalette.ColorRole.PlaceholderText, "#74889e"),
    ):
        palette.setColor(role, QColor(value))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6e8093"))
    application.setPalette(palette)
    application.setStyleSheet(STYLE)
    application.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "application.png")))


@lru_cache(maxsize=128)
def icon(name, color="#a3b8cc", size=20):
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 24, size / 24)
    painter.setPen(
        QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    )
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if name in ("brand", "frame"):
        for x, y, dx, dy in ((4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)):
            painter.drawLine(QPointF(x, y + 5 * dy), QPointF(x, y))
            painter.drawLine(QPointF(x, y), QPointF(x + 5 * dx, y))
        if name == "brand":
            painter.setBrush(QColor(color))
            painter.drawEllipse(QRectF(9, 9, 6, 6))
    elif name == "target":
        painter.drawEllipse(QRectF(4, 4, 16, 16))
        painter.drawEllipse(QRectF(9, 9, 6, 6))
        for a, b in (((12, 1), (12, 5)), ((12, 19), (12, 23)), ((1, 12), (5, 12)), ((19, 12), (23, 12))):
            painter.drawLine(QPointF(*a), QPointF(*b))
    elif name in ("source", "monitor", "output"):
        painter.drawRoundedRect(QRectF(3, 4, 18, 13), 2, 2)
        painter.drawLine(QPointF(12, 17), QPointF(12, 21))
        painter.drawLine(QPointF(8, 21), QPointF(16, 21))
        if name == "source":
            painter.drawLine(QPointF(9, 8), QPointF(15, 11))
            painter.drawLine(QPointF(15, 11), QPointF(9, 14))
    elif name == "folder":
        path = QPainterPath(QPointF(3, 8))
        for point in ((3, 5), (9, 5), (11, 8), (21, 8), (21, 19), (3, 19), (3, 8)):
            path.lineTo(*point)
        painter.drawPath(path)
    elif name == "camera":
        painter.drawRoundedRect(QRectF(3, 6, 14, 13), 2, 2)
        path = QPainterPath(QPointF(17, 10))
        for point in ((22, 7), (22, 18), (17, 15)):
            path.lineTo(*point)
        painter.drawPath(path)
    elif name in ("play", "pause"):
        if name == "play":
            path = QPainterPath(QPointF(8, 5))
            path.lineTo(19, 12)
            path.lineTo(8, 19)
            path.closeSubpath()
            painter.setBrush(QColor(color))
            painter.drawPath(path)
        else:
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(7, 5, 3, 14), 1, 1)
            painter.drawRoundedRect(QRectF(14, 5, 3, 14), 1, 1)
    elif name == "export":
        painter.drawLine(QPointF(12, 15), QPointF(12, 3))
        painter.drawLine(QPointF(7, 8), QPointF(12, 3))
        painter.drawLine(QPointF(12, 3), QPointF(17, 8))
        painter.drawLine(QPointF(4, 14), QPointF(4, 21))
        painter.drawLine(QPointF(4, 21), QPointF(20, 21))
        painter.drawLine(QPointF(20, 21), QPointF(20, 14))
    elif name in ("chevron", "expand"):
        painter.drawLine(QPointF(9, 6), QPointF(15, 12))
        painter.drawLine(QPointF(15, 12), QPointF(9, 18))
    elif name == "down":
        painter.drawLine(QPointF(6, 9), QPointF(12, 15))
        painter.drawLine(QPointF(12, 15), QPointF(18, 9))
    elif name == "restart":
        painter.drawArc(QRectF(5, 5, 14, 14), 30 * 16, 290 * 16)
        painter.drawLine(QPointF(4, 3), QPointF(4, 9))
        painter.drawLine(QPointF(4, 9), QPointF(10, 9))
    elif name == "refresh":
        painter.drawArc(QRectF(5, 5, 14, 14), 40 * 16, 290 * 16)
        painter.drawLine(QPointF(19, 3), QPointF(19, 9))
        painter.drawLine(QPointF(19, 9), QPointF(13, 9))
    elif name == "layers":
        for y in (7, 12, 17):
            path = QPainterPath(QPointF(3, y))
            path.lineTo(12, y + 4)
            path.lineTo(21, y)
            painter.drawPath(path)
    elif name == "more":
        painter.setBrush(QColor(color))
        for x in (5, 12, 19):
            painter.drawEllipse(QRectF(x - 1, 11, 2, 2))
    elif name == "black":
        painter.drawRoundedRect(QRectF(4, 4, 16, 16), 3, 3)
        painter.drawLine(QPointF(7, 17), QPointF(17, 7))
    painter.end()
    return QIcon(pixmap)


def paint_empty_preview(painter, rect, role, title, detail):
    painter.fillRect(rect, QColor("#0c131d"))
    center = rect.center()
    y = center.y() - 30
    painter.setPen(QPen(QColor("#263d4c"), 1))
    painter.setBrush(QColor("#111f2b"))
    painter.drawRoundedRect(QRectF(center.x() - 28, y - 28, 56, 56), 15, 15)
    icon({"input": "source", "output": "frame", "target": "target"}.get(role, "frame"), "#73b6ac", 28).paint(
        painter, round(center.x() - 14), round(y - 14), 28, 28
    )
    painter.setPen(QColor("#becddd"))
    font = painter.font()
    font.setPointSizeF(11)
    font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(font)
    painter.drawText(QRectF(18, y + 38, rect.width() - 36, 24), Qt.AlignmentFlag.AlignHCenter, title)
    painter.setPen(QColor("#7f96ad"))
    font.setPointSizeF(9)
    font.setWeight(QFont.Weight.Normal)
    painter.setFont(font)
    painter.drawText(
        QRectF(20, y + 65, rect.width() - 40, 40),
        Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
        detail,
    )
