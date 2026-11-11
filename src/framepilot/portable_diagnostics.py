"""File/photo/UI/MP4 smoke test shared by Windows and macOS; no OS capture claim."""

import json
import os
import platform
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from PySide6.QtCore import QTimer

from . import __version__


def portable_smoke_test(app, report):
    from .app import MainWindow
    from .exporter import ExportJob
    from .geometry import Rect
    from .pipeline import Settings

    temporary = tempfile.TemporaryDirectory(prefix="framepilot-portable-")
    os.environ["FRAMEPILOT_CONFIG_DIR"] = temporary.name
    window = MainWindow()
    window.show()
    window.preferred_port = 0
    window.start_local_output()
    window.load_photo(Path(__file__).parent / "assets/astronaut.png")
    phase = 0
    started = time.monotonic()
    timer = QTimer(window)
    result = {
        "version": __version__,
        "platform": platform.system(),
        "machine": platform.machine(),
        "passed": False,
        "checks": {},
        "window_capture": "not_verified",
        "physical_camera": "not_verified",
        "obs_virtual_camera": "not_verified",
    }

    def finish(error=None):
        timer.stop()
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        if error:
            result["error"] = str(error)
        window.close()
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        app.exit(0 if result["passed"] else 1)

    def poll():
        nonlocal phase
        try:
            if time.monotonic() - started > 45:
                raise RuntimeError("Timeout: " + window.status.text())
            if phase == 0 and window.photo_ready and not window.busy:
                window.photo_face()
                phase = 1
            elif (
                phase == 1
                and not window.busy
                and window.input_view.boxes
                and window.input_view.boxes[0].area < 0.2
            ):
                result["checks"]["face_backend"] = window.video_worker.pipeline.detector.backend
                window.save_photo_image(Path(temporary.name) / "photo.png")
                result["checks"]["photo_face_and_save"] = True
                window.manual_frame(Rect(0.2, 0.1, 0.6, 0.8))
                phase = 2
            elif phase == 2 and not window.busy and not window.render_pending:
                with urlopen(
                    window.publisher.url.replace("/framepilot", "/stream.mjpg"), timeout=3
                ) as response:
                    response.readline()
                    response.readline()
                    length = int(response.readline().decode().split(":")[1])
                    response.readline()
                    image = cv2.imdecode(np.frombuffer(response.read(length), np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.max() < 100:
                    raise RuntimeError("Local video output is blank")
                result["checks"]["local_mjpeg_and_frame"] = True
                portrait = cv2.imread(str(Path(__file__).parent / "assets/astronaut.png"))
                path = Path(temporary.name) / "clip.avi"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
                if not writer.isOpened():
                    raise RuntimeError("Video fixture encoder is unavailable")
                for _ in range(12):
                    writer.write(cv2.resize(portrait, (320, 240)))
                writer.release()
                window.load_video(path)
                phase = 3
            elif (
                phase == 3
                and not window.busy
                and window.latest_frame is not None
                and window.latest_media_time > 0.15
            ):
                window.pause_source()
                result["checks"]["qt_video_playback"] = True
                window.begin_export(
                    ExportJob(
                        window.video_path,
                        Path(temporary.name) / "result.mp4",
                        Settings(width=160, height=96, loss="original"),
                        fps=15,
                        audio=False,
                    )
                )
                phase = 4
            elif phase == 4 and window.export_result is not None:
                if window.export_result.get("cancelled"):
                    raise RuntimeError("MP4 export failed: " + window.export_status.text())
                result["checks"]["mp4_export_frames"] = window.export_result["frames"]
                result["passed"] = True
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(poll)
    timer.start(40)
    code = app.exec()
    temporary.cleanup()
    return code
