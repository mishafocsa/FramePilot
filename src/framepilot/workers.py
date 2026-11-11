"""Separate serial workers: screen permission/capture cannot block video processing."""

import threading
import time

from PySide6.QtCore import QObject, Signal, Slot

from .pipeline import Pipeline
from .vision import detect_slots


class VideoWorker(QObject):
    completed = Signal(object)
    failed = Signal(int, str)

    def __init__(self, publisher=None):
        super().__init__()
        self.pipeline = None
        self.publisher = publisher
        self.last_result = self.last_error = None

    @Slot(object)
    def drain(self, receipt):
        receipt.update(result=self.last_result, error=self.last_error)
        receipt["ready"].set()

    @Slot(object)
    def process(self, job):
        session, image, settings, click, reset, *options = job
        self.last_result = self.last_error = None
        try:
            if self.pipeline is None:
                self.pipeline = Pipeline()
            start = time.perf_counter()
            repaint, media_time = options[:2] if options else (False, start)
            submitted_at = options[2] if len(options) > 2 else time.monotonic()
            timeout = options[3] if len(options) > 3 else None
            output, face, faces = self.pipeline.process(
                image,
                media_time,
                settings,
                click,
                reset,
                repaint,
                render_time=time.monotonic(),
                source_opacity=options[4] if len(options) > 4 else None,
                still=options[5] if len(options) > 5 else False,
            )
            if self.publisher is not None:
                deadline = submitted_at + timeout if timeout is not None and not repaint else None
                self.publisher.publish(output, session, deadline=deadline)
            active = settings.active_slot or (settings.slots[0].identifier if settings.slots else "single")
            self.last_result = (
                session,
                output,
                face,
                faces,
                (time.perf_counter() - start) * 1000,
                "OpenCV · движение и образец предмета"
                if settings.subject_mode == "object"
                else self.pipeline.detector.backend,
                self.pipeline.last_eyes,
                {
                    "tracks": self.pipeline.last_tracks,
                    "recovery": self.pipeline.last_recovery,
                    "tracking": self.pipeline.last_quality,
                    "composition": self.pipeline.last_tracks.get(active, {}).get("composition", {}),
                    "surface": self.pipeline.last_tracks.get(active, {}).get("surface"),
                    "time": media_time,
                    "repaint": repaint,
                    "submitted_at": submitted_at,
                    "framing": self.pipeline.last_framing,
                    "animating": any(
                        info.get("framing", {}).get("transitioning", False)
                        for info in self.pipeline.last_tracks.values()
                    ),
                },
            )
            self.completed.emit(self.last_result)
        except Exception as error:
            self.last_error = (session, str(error))
            self.failed.emit(session, str(error))


class CaptureWorker(QObject):
    windows_ready = Signal(object)
    scanned = Signal(object)
    failed = Signal(int, str)

    def __init__(self):
        super().__init__()
        self.capture = None

    @Slot()
    def list_windows(self):
        try:
            if self.capture is None:
                from .capture import create_capture

                self.capture = create_capture()
            self.windows_ready.emit(self.capture.windows())
        except Exception as error:
            self.failed.emit(-1, str(error))

    @Slot(object)
    def scan(self, job):
        generation, window, internal_image, *options = job
        try:
            if internal_image is not None:
                image = internal_image
            else:
                if self.capture is None:
                    from .capture import create_capture

                    self.capture = create_capture()
                image = self.capture.capture(window)
            roi, sensitivity = options if options else (None, "normal")
            self.scanned.emit((generation, image, detect_slots(image, roi, sensitivity)))
        except Exception as error:
            self.failed.emit(generation, str(error))

    @Slot()
    def close(self):
        if self.capture is not None and hasattr(self.capture, "close"):
            self.capture.close()


class OBSWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.cancel = threading.Event()
        self.client = None

    @Slot()
    def disconnect(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    @Slot(object)
    def configure(self, job):
        from .obs import OBSActionError, OBSClient

        kind, payload = job
        try:
            if kind == "configure":
                self.disconnect()
                port, password, url, width, height, activate, fps = payload
                self.client = OBSClient(port, password, self.cancel)
                self.client.configure(url, width, height, activate, keep_open=True, fps=fps)
            elif kind == "disconnect":
                self.disconnect()
                self.completed.emit({"connected": False, "message": "OBS отключена"})
                return
            elif self.client is None:
                raise RuntimeError("Нет подключения к OBS")
            elif kind == "sync":
                self.client.sync(*payload)
            elif kind == "diagnose":
                self.completed.emit(self.client.diagnose())
                return
            state = self.client.status()
            if kind == "configure":
                state["synced"] = (url, width, height, fps)
            elif kind == "sync":
                state["synced"] = payload
            self.completed.emit(state)
        except OBSActionError as error:
            self.disconnect()
            self.failed.emit(str(error))
        except Exception:
            self.disconnect()
            # Connection exceptions may contain transport details. Never echo the password.
            self.failed.emit(
                "Не удалось настроить OBS. Проверьте, что OBS открыт, WebSocket включён, "
                "порт и пароль верны. Имена «FramePilot»/«FramePilot Video» должны быть свободны "
                "или принадлежать FramePilot."
            )


class ExportWorker(QObject):
    progress = Signal(int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.cancel = threading.Event()

    @Slot(object)
    def export(self, job):
        from .exporter import ExportCancelled, export_video

        try:
            count = export_video(job, self.cancel, self.progress.emit)
            self.completed.emit({"cancelled": False, "frames": count, "path": str(job.destination)})
        except ExportCancelled:
            self.completed.emit({"cancelled": True})
        except Exception as error:
            self.failed.emit(str(error))
