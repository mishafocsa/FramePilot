"""End-to-end smoke test, usable from the frozen .app without Python installed."""

import json
import os
import platform
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QTimer

from . import __version__

FORMAT_CASES = (
    ("mp4", "libx264", ()),
    ("mov", "libx265", ("-x265-params", "pools=1:frame-threads=1:log-level=error")),
    ("mov", "prores_ks", ("-profile:v", "2")),
    ("mkv", "mpeg4", ()),
    ("avi", "mjpeg", ()),
    ("webm", "libvpx-vp9", ("-deadline", "realtime")),
    ("mp4", "libaom-av1", ("-cpu-used", "8", "-crf", "40")),
)


def surface_rotation_check():
    from .geometry import Rect
    from .pipeline import Pipeline, Settings
    from .tracking_validation import rigid_frame

    pipeline = Pipeline()
    settings = Settings(
        width=320,
        height=240,
        subject_mode="object",
        surface_lock=True,
        subject_region=Rect(0, 0, 0.5, 1),
        padding=0,
    )
    maximum = 0
    for i in range(45):
        image, box = rigid_frame(i, angle=i * 2)
        output, subject, _ = pipeline.process(image, i / 30, settings, box if i == 0 else None, i == 0)
        if subject is None or not pipeline.renderer.last_surface_quad:
            raise RuntimeError("Не подтверждена геометрия при повороте")
        excluded = int(((output[:, :, 1] > 205) & (output[:, :, 2] < 120) & (output[:, :, 0] < 100)).sum())
        maximum = max(maximum, excluded)
        if excluded > 15:
            raise RuntimeError("Исключённая часть попала в выход при повороте")
    return {
        "frames": 45,
        "rotation_degrees": 88,
        "maximum_excluded_marker_pixels": maximum,
        "scope": "Synthetic rigid 2-D card; no 3-D or arbitrary-object accuracy claim",
    }


def smoke_test(app, report: Path) -> int:
    from .app import MainWindow
    from .exporter import ExportJob
    from .geometry import Rect
    from .obs import authentication

    started = time.monotonic()
    temporary = tempfile.TemporaryDirectory(prefix="framepilot-check-")
    os.environ["FRAMEPILOT_CONFIG_DIR"] = temporary.name
    window = MainWindow()
    window.show()
    window.show_demo()
    window.demo.animate = False
    window.load_sample()
    window.align_eyes.setChecked(True)
    window.preferred_port = 0
    window.start_local_output()
    timer = QTimer(window)
    outcome = {
        "application": "FramePilot",
        "version": __version__,
        "machine": platform.machine(),
        "passed": False,
        "checks": {},
        "elapsed_seconds": 0,
    }
    phase = {"export_started": False, "scene_started": False, "object_started": False, "adaptive": 0}
    connection = {"wizard": None}

    def finish(error=None):
        timer.stop()
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 2)
        if error:
            outcome["error"] = str(error)
            outcome["phase"] = {key: value for key, value in phase.items() if key != "tracked_box"}
            if connection["wizard"] is not None:
                outcome["connection_diagnostics"] = connection["wizard"].report()
        try:
            report.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(report.with_suffix(".png")))
            report.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            if connection["wizard"] is not None:
                connection["wizard"].reject()
            window.close()
            app.exit(0 if outcome["passed"] else 1)

    def poll():
        try:
            if time.monotonic() - started > 90:
                finish("Таймаут: " + window.status.text())
                return
            if "Ошибка" in window.status.text():
                finish(window.status.text())
                return
            if "photo" in phase:
                import cv2
                import numpy as np

                stage = phase["photo"]
                if stage == 0:
                    png = Path(temporary.name) / "transparent.png"
                    rgba = np.full((120, 200, 4), (20, 60, 200, 128), np.uint8)
                    if not cv2.imwrite(str(png), rgba):
                        raise RuntimeError("Не подготовлено фото PNG")
                    window.load_photo(png)
                    phase["photo"] = 1
                elif stage == 1 and window.photo_ready and not window.busy:
                    from .widgets import to_array

                    pixel = to_array(window.last_output)[
                        window.settings.height // 2, window.settings.width // 2
                    ]
                    if not np.allclose(pixel, (10, 30, 100), atol=2):
                        raise RuntimeError("Неверное смешивание прозрачности PNG")
                    for suffix in ("png", "jpg"):
                        path = Path(temporary.name) / ("photo-output." + suffix)
                        window.save_photo_image(path)
                        if cv2.imread(str(path)).shape[:2] != (window.settings.height, window.settings.width):
                            raise RuntimeError("Неверный размер сохранённого фото")
                    jpg = Path(temporary.name) / "portrait.jpg"
                    portrait = cv2.imread(str(Path(__file__).parent / "assets/astronaut.png"))
                    cv2.imwrite(str(jpg), portrait)
                    window.load_photo(jpg)
                    phase["photo"] = 2
                elif stage == 2 and window.photo_ready and not window.busy:
                    window.photo_face()
                    phase["photo"] = 3
                elif (
                    stage == 3
                    and not window.busy
                    and window.input_view.boxes
                    and window.input_view.boxes[0].area < 0.2
                ):
                    window.manual_frame(Rect(0.2, 0.1, 0.5, 0.7))
                    window.fps_combo.setCurrentIndex(2)
                    phase["photo"] = 4
                elif stage == 4 and not window.busy and not window.render_pending:
                    window.studio.begin_recording(Path(temporary.name) / "photo-recording.mp4")
                    phase["photo_record_started"] = time.monotonic()
                    phase["photo"] = 5
                elif stage == 5 and time.monotonic() - phase["photo_record_started"] > 0.5:
                    window.studio.stop_recording()
                    phase["photo"] = 6
                elif stage == 6 and not window.studio.recorder.active:
                    if window.studio.recorder.error:
                        raise RuntimeError(window.studio.recorder.error)
                    cap = cv2.VideoCapture(str(Path(temporary.name) / "photo-recording.mp4"))
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if count < 5:
                        raise RuntimeError("Не записано статичное фото в MP4")
                    outcome["checks"]["photos"] = {
                        "jpeg_face": True,
                        "png_alpha": True,
                        "saved_png_jpeg": True,
                        "static_recorded_frames": count,
                        "dynamic_frame": True,
                    }
                    outcome["passed"] = True
                    finish()
                return
            if "formats" in phase:
                import imageio_ffmpeg

                from .pipeline import Settings

                index = phase["formats"]
                if index == len(FORMAT_CASES):
                    if not phase.get("batch_test"):
                        phase["batch_test"] = True
                        for entry in FORMAT_CASES[:2]:
                            source = Path(temporary.name) / (entry[1] + "." + entry[0])
                            window.studio.batch.add(
                                ExportJob(
                                    source,
                                    Path(temporary.name) / (entry[1] + "-queue.mp4"),
                                    Settings(width=160, height=96, loss="original"),
                                    fps=15,
                                    audio=False,
                                )
                            )
                        window.studio.batch.start()
                        return
                    if window.studio.batch.running or window.export_busy:
                        return
                    if any(e.state != "done" for e in window.studio.batch.entries):
                        raise RuntimeError("Не выполнена очередь экспорта")
                    outcome["checks"]["batch_exports"] = len(window.studio.batch.entries)
                    if not phase.get("recovery_test"):
                        phase["recovery_test"] = True
                        window.studio.autosave()
                        return
                    if window.studio.recovery.busy:
                        return
                    from .recovery import read_recovery

                    data = read_recovery(window.studio.recovery.path)
                    if data is None or Path(data["source"]) != window.video_path:
                        raise RuntimeError("Не сохранён файловый сеанс")
                    outcome["checks"]["file_session_autosave"] = True
                    outcome["checks"]["surface_rotation"] = surface_rotation_check()
                    window.remove_background.setChecked(False)
                    window.mode.setCurrentIndex(0)
                    while window.scene_slots:
                        window.remove_scene_slot()
                    phase["photo"] = 0
                    return
                extension, codec, extra = FORMAT_CASES[index]
                source = Path(temporary.name) / (codec + "." + extension)
                if not phase.get("format_loaded"):
                    subprocess.run(
                        [
                            imageio_ffmpeg.get_ffmpeg_exe(),
                            "-nostdin",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "lavfi",
                            "-i",
                            "testsrc2=size=160x96:rate=15",
                            "-t",
                            "0.4",
                            "-c:v",
                            codec,
                            "-threads",
                            "1",
                            *extra,
                            str(source),
                        ],
                        capture_output=True,
                        check=True,
                        timeout=20,
                    )
                    window.load_video(source)
                    window.subject_mode.setCurrentIndex(0)
                    phase["format_loaded"] = True
                    phase["format_export"] = False
                    return
                if not window.studio.media.preview or window.studio.media.result is None or window.busy:
                    if getattr(window.studio.media, "error", None):
                        raise RuntimeError(window.studio.media.error)
                    return
                window.pause_source()
                if not phase["format_export"]:
                    phase["format_export"] = True
                    window.begin_export(
                        ExportJob(
                            source,
                            Path(temporary.name) / (codec + "-export.mp4"),
                            Settings(width=160, height=96, loss="original"),
                            fps=15,
                            audio=False,
                        )
                    )
                    return
                if window.export_busy:
                    return
                if (
                    not window.export_result
                    or window.export_result.get("cancelled")
                    or window.export_result.get("frames") != 6
                ):
                    raise RuntimeError("Формат не прошёл экспорт: " + codec)
                outcome["checks"].setdefault("format_matrix", []).append(
                    {
                        "container": extension,
                        "codec": window.studio.media.result["codec"],
                        "preview": True,
                        "ffmpeg_first_frame": True,
                        "export_frames": 6,
                    }
                )
                phase["formats"] += 1
                phase["format_loaded"] = False
                return
            if connection["wizard"] is not None:
                wizard = connection["wizard"]
                if len(wizard.history.rows) < 3:
                    return
                outcome["checks"]["connection_wizard_internal_roundtrip"] = wizard.report()
                if wizard.history.stable_calibration() is None:
                    raise RuntimeError("Нестабильная геометрия проверочной сетки")
                wizard.reject()
                connection["wizard"] = None
                phase["formats"] = 0
                phase["format_loaded"] = False
                return
            if phase.get("studio", 0):
                import cv2
                import imageio_ffmpeg
                import numpy as np

                from .screen_source import receiver_class

                stage = phase["studio"]
                if stage == 1:
                    if window.studio.analysis_busy or window.busy:
                        return
                    tracks = window.settings.automation.tracks
                    if not tracks or len(tracks[0].samples) < 30:
                        raise RuntimeError(
                            "Анализ ролика не выдал рассчитанную траекторию: "
                            + window.studio.analysis_status.text()
                        )
                    outcome["checks"]["clip_analysis"] = {
                        "targets": len(tracks),
                        "samples": len(tracks[0].samples),
                        "start": tracks[0].start,
                        "fps": tracks[0].fps,
                        "synthetic_fixture": True,
                    }
                    project = Path(temporary.name) / "clip.framepilot.json"
                    window.studio.save_project(project)
                    window.studio.load_project(project)
                    if window.settings.automation.tracks != tracks:
                        raise RuntimeError("Проект не сохранил траекторию")
                    outcome["checks"]["clip_project_roundtrip"] = True
                    alpha = np.zeros((64, 64, 4), np.uint8)
                    alpha[8:56, 8:56, 3] = 255
                    mask_path = Path(temporary.name) / "frame-mask.png"
                    cv2.imwrite(str(mask_path), alpha)
                    window.studio.mask_path = str(mask_path)
                    window.studio.mask_kind.setCurrentIndex(window.studio.mask_kind.findData("png"))
                    window.mode.setCurrentIndex(0)
                    window.begin_export(
                        ExportJob(
                            window.video_path,
                            Path(temporary.name) / "cached-mask.mp4",
                            replace(window.settings, width=240, height=160),
                            fps=15,
                            start=tracks[0].start,
                            audio=False,
                        )
                    )
                    window.studio.begin_recording(Path(temporary.name) / "live.mp4")
                    phase["studio"] = 2
                    return
                if stage == 2:
                    if window.studio.recorder.frames < 5:
                        return
                    window.toggle_black_output()
                    phase["studio"] = 3
                    return
                if stage == 3:
                    if window.studio.recorder.frames < 12:
                        return
                    window.studio.stop_recording()
                    phase["studio"] = 4
                    return
                if window.studio.recorder.active or window.export_busy:
                    return
                if (
                    window.studio.recorder.error
                    or not window.export_result
                    or window.export_result.get("cancelled")
                ):
                    raise RuntimeError(window.studio.recorder.error or window.export_status.text())
                video = imageio_ffmpeg.read_frames(str(Path(temporary.name) / "live.mp4"), pix_fmt="bgr24")
                recording_metadata = next(video)
                recording_images = [np.frombuffer(data, np.uint8) for data in video]
                video.close()
                if (
                    len(recording_images) < 12
                    or recording_images[-1].max() > 8
                    or not recording_images[0].any()
                ):
                    raise RuntimeError("Запись не сохранила видео и ручной чёрный кадр")
                outcome["checks"]["live_output_recording"] = {
                    "frames": len(recording_images),
                    "fps": recording_metadata["fps"],
                    "size": recording_metadata["size"],
                    "black_retained": True,
                    "audio": False,
                }
                video = imageio_ffmpeg.read_frames(
                    str(Path(temporary.name) / "cached-mask.mp4"), pix_fmt="bgr24"
                )
                mask_metadata = next(video)
                masked = np.frombuffer(next(video), np.uint8).reshape(160, 240, 3)
                video.close()
                if masked[0, 0].max() > 8 or not masked[40:120, 60:180].any():
                    raise RuntimeError("Маска не применилась к экспорту рассчитанной траектории")
                outcome["checks"]["analysed_trajectory_png_mask_export"] = {
                    "size": mask_metadata["size"],
                    "codec": mask_metadata["codec"],
                    "selection_not_required": True,
                }
                outcome["checks"]["screen_stream_api_bound"] = receiver_class().alloc().init() is not None
                outcome["checks"]["real_screen_stream_verified"] = False
                window.toggle_black_output()
                window.studio.mask_kind.setCurrentIndex(0)
                from .connections import ConnectionWizard

                connection["wizard"] = ConnectionWizard(window)
                connection["wizard"].show()
                connection["wizard"].start_probe()
                return
            if phase["object_started"]:
                pipeline = window.video_worker.pipeline
                if window.busy:
                    return
                if phase["adaptive"] == 0:
                    if not window.input_view.focus or pipeline.object_tracker.flow_updates < 10:
                        return
                    window.pause_source()
                    outcome["checks"]["whole_object_tracking_with_partial_output"] = {
                        "synthetic_fixture": True,
                        "region": list(window.capture_profile()["subject_region"]),
                        "flow_updates": pipeline.object_tracker.flow_updates,
                        "template_updates": pipeline.object_tracker.template_updates,
                        "motion_mode": window.settings.motion_mode,
                        "target_scan_interval_ms": window.scan_timer.interval(),
                    }
                    phase["tracked_box"] = pipeline.object_tracker.box
                    evidence = window.last_tracking_quality
                    if evidence.get("state") not in ("selected", "stable", "weak"):
                        raise RuntimeError("Нет данных контроля качества выбранного предмета")
                    outcome["checks"]["object_tracking_evidence"] = dict(evidence)
                    window.manual_frame(Rect(0.1, 0.1, 0.2, 0.4))
                    window.auto_framing_enabled.setChecked(True)
                    phase["adaptive"] = 1
                    return
                if phase["adaptive"] == 1:
                    if not window.last_framing.get("enabled") or window.last_framing.get("whole"):
                        return
                    window.manual_frame(Rect(0.1, 0.1, 0.8, 0.8))
                    phase["adaptive"] = 2
                    return
                if phase["adaptive"] == 2:
                    if not window.last_framing.get("whole") or window.last_framing.get("transitioning"):
                        return
                    window.manual_frame(Rect(0.1, 0.1, 0.2, 0.4))
                    phase["adaptive"] = 3
                    return
                if window.last_framing.get("whole") or window.last_framing.get("transitioning"):
                    return
                if pipeline.object_tracker.box != phase["tracked_box"]:
                    raise RuntimeError("Автоматическое кадрирование изменило выбранную цель")
                outcome["checks"]["automatic_part_whole_return"] = {
                    "while_paused": True,
                    "same_target": True,
                    "completed": True,
                }
                import cv2
                import numpy as np

                from .widgets import to_array

                window.toggle_black_output()
                # A late publisher must not override the deliberately closed output.
                window.publisher.publish(to_array(window.last_normal_output), window.session)
                held = cv2.imdecode(np.frombuffer(window.publisher.jpeg, np.uint8), cv2.IMREAD_COLOR)
                if held is None or held.any() or to_array(window.last_output).any():
                    raise RuntimeError("Ручной чёрный кадр не удержался в выходе")
                with urllib.request.urlopen(
                    window.publisher.url.replace("/framepilot", "/api/status"), timeout=2
                ) as response:
                    health = json.load(response)
                if health.get("hold_reason") != "emergency_black":
                    raise RuntimeError("HTTP-статус не отражает закрытый выход")
                window.toggle_black_output()
                if window.publisher.hold_reason is not None or not to_array(window.last_output).any():
                    raise RuntimeError("Видео не восстановилось после ручного чёрного кадра")
                if pipeline.object_tracker.box != phase["tracked_box"]:
                    raise RuntimeError("Закрытие выхода сбросило выбранную цель")
                outcome["checks"]["manual_black_and_restore"] = {
                    "local_preview": True,
                    "local_mjpeg": True,
                    "http_status": True,
                    "same_tracked_target": True,
                }
                window.auto_framing_enabled.setChecked(False)
                window.studio.add_cue()
                window.studio.start_analysis()
                phase["studio"] = 1
                return
            if (
                window.output_view.image.isNull()
                or not window.input_view.boxes
                or (not phase["scene_started"] and window.stable.accepted is None)
            ):
                return
            if window.output_view.image.size().width() != window.settings.width:
                return
            with urllib.request.urlopen(
                window.publisher.url.replace("/framepilot", "/api/status"), timeout=2
            ) as response:
                status = json.load(response)
            if status.get("frames", 0) < 1:
                return
            outcome["checks"].update(
                {
                    "video_playback": True,
                    "native_face_detection": True,
                    "automatic_slot_detection": True,
                    "output_image": True,
                    "local_output": status,
                    "screen_capture_api": False,
                    "obs_client_available": bool(authentication("test", "salt", "challenge")),
                    "native_eye_landmarks": window.video_worker.pipeline.last_eyes is not None,
                }
            )
            if not outcome["checks"]["native_eye_landmarks"]:
                return
            if platform.system() == "Darwin":
                from .capture import MacCapture

                capture = MacCapture()
                outcome["checks"]["screen_capture_api"] = hasattr(
                    capture.sc.SCScreenshotManager, "captureImageWithFilter_configuration_completionHandler_"
                )
            if not phase["scene_started"]:
                phase["scene_started"] = True
                window.pause_source()
                window.add_scene_slot()
                window.subject_mode.setCurrentIndex(window.subject_mode.findData("upper"))
                window.scene_combo.setCurrentIndex(0)
                window.remove_background.setChecked(True)
                return
            pipeline = window.video_worker.pipeline
            if len(window.last_tracks) != 2 or window.busy:
                return
            if not all(info["subject"] is not None for info in window.last_tracks.values()):
                return
            if pipeline.mask_cache is None:
                return
            outcome["checks"]["independent_scene_slots"] = len(window.last_tracks)
            outcome["checks"]["native_upper_body"] = any(
                info["mode"] == "upper" for info in window.last_tracks.values()
            )
            outcome["checks"]["native_person_segmentation"] = {
                "minimum": round(float(pipeline.mask_cache.min()), 3),
                "maximum": round(float(pipeline.mask_cache.max()), 3),
            }
            outcome["checks"]["camera_input_available"] = callable(window.camera_input.start)
            if not phase["export_started"]:
                phase["export_started"] = True
                window.save_scenario("Smoke scene")
                outcome["checks"]["scenario_saved"] = bool(window.preferences.load().get("scenarios"))
                window.begin_export(
                    ExportJob(
                        window.video_path,
                        Path(temporary.name) / "smoke.mp4",
                        replace(window.settings, width=320, height=240),
                        fps=15,
                        audio=False,
                        selection=window.current_selections(),
                    )
                )
                return
            if window.export_busy:
                return
            if not window.export_result or window.export_result.get("cancelled"):
                raise RuntimeError(window.export_status.text())
            import imageio_ffmpeg

            frames = imageio_ffmpeg.read_frames(str(Path(temporary.name) / "smoke.mp4"))
            metadata = next(frames)
            frames.close()
            outcome["checks"]["mp4_export"] = {
                "size": metadata["size"],
                "codec": metadata["codec"],
                "frames": window.export_result["frames"],
                "fps": metadata["fps"],
            }
            if metadata["size"] != (320, 240) or "h264" not in metadata["codec"]:
                raise RuntimeError("Экспортированный файл не соответствует выбранным параметрам")
            phase["object_started"] = True
            profile = window.capture_profile()
            profile.update(
                slots=[],
                active_slot="",
                remove_background=False,
                subject_mode="object",
                subject_region=[0, 0, 0.5, 1],
                motion_mode="responsive",
            )
            window.apply_profile(profile)
            window.load_object_sample()
        except Exception as error:
            finish(error)

    timer.timeout.connect(poll)
    timer.start(100)
    result = app.exec()
    temporary.cleanup()
    return result
