"""Qt camera input, activated only by an explicit user action."""

from PySide6.QtCore import QCameraPermission, QObject, Qt, Signal
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import QApplication


class CameraInput(QObject):
    frame = Signal(object)
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = QMediaCaptureSession(self)
        self.sink = None
        self.camera = None
        self.paused = False
        self.generation = 0
        self.device = None

    @staticmethod
    def devices():
        return list(QMediaDevices.videoInputs())

    def start(self, device):
        self.stop()
        token = self.generation
        self.device = device
        application = QApplication.instance()
        permission = QCameraPermission()
        status = application.checkPermission(permission)
        if status == Qt.PermissionStatus.Granted:
            self._open(device, token)
        elif status == Qt.PermissionStatus.Denied:
            self.message.emit("Разрешите FramePilot доступ к камере в настройках системы")
        else:
            application.requestPermission(
                permission,
                self,
                lambda result: self._permission_result(result, device, token),
            )

    def _permission_result(self, result, device, token):
        if token != self.generation:
            return
        if result.status() == Qt.PermissionStatus.Granted:
            self._open(device, token)
        else:
            self.message.emit("Доступ к камере не разрешён")

    def _open(self, device, token):
        if token != self.generation:
            return
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(
            lambda frame: self.frame.emit(frame) if token == self.generation else None
        )
        self.session.setVideoSink(self.sink)
        self.camera = QCamera(device, self)
        self.camera.errorOccurred.connect(
            lambda error, text: (
                self.message.emit("Ошибка камеры: " + text) if token == self.generation else None
            )
        )
        self.session.setCamera(self.camera)
        self.paused = False
        self.camera.start()
        if token == self.generation:
            self.message.emit("Камера включается…")

    def toggle_pause(self):
        if self.camera is None:
            return
        self.paused = not self.paused
        if self.paused:
            self.camera.stop()
            self.message.emit("Камера на паузе · устройство остановлено")
        else:
            self.camera.start()

    def stop(self):
        self.generation += 1
        if self.camera is not None:
            self.camera.stop()
            self.session.setCamera(None)
            self.camera.deleteLater()
            self.camera = None
        if self.sink is not None:
            self.session.setVideoSink(None)
            self.sink.deleteLater()
            self.sink = None
        self.paused = False
