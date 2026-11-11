"""Validated JSON preferences. No passwords, video frames or captured windows are persisted."""

import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

from .framing import FramingRule
from .geometry import Rect
from .masks import FrameMask
from .object_geometry import QuadSelection, valid_quad


def validate_profile(data: object) -> dict:
    if not isinstance(data, dict) or data.get("version") not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        raise ValueError("Неподдерживаемая версия профиля")
    profile = dict(data)
    profile.setdefault("surface_lock", False)
    if type(profile["surface_lock"]) is not bool:
        raise ValueError("Неверная привязка к поверхности")
    profile["frame_mask"] = FrameMask.from_record(profile.get("frame_mask")).record()
    profile.setdefault("framing_zoom", 1.0)
    if (
        type(profile["framing_zoom"]) not in (int, float)
        or not math.isfinite(profile["framing_zoom"])
        or not 0.5 <= profile["framing_zoom"] <= 2
    ):
        raise ValueError("Неверный масштаб кадра")
    for key, low, high in [("padding", 0, 1), ("response", 0.05, 2)]:
        value = profile.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Некорректное поле {key}")
    for key, count in [("fill", 2), ("loss", 3), ("mapping", 2), ("resolution", 3), ("fps", 3)]:
        if type(profile.get(key)) is not int or not 0 <= profile[key] < count:
            raise ValueError(f"Некорректное поле {key}")
    if type(profile.get("auto_aspect")) is not bool:
        raise ValueError("Некорректное поле auto_aspect")
    profile.setdefault("sensitivity", "normal")
    if profile["sensitivity"] not in ("normal", "sensitive"):
        raise ValueError("Некорректная чувствительность поиска рамок")
    profile.setdefault("subject_mode", "face")
    if profile["subject_mode"] not in ("face", "object", "upper", "person"):
        raise ValueError("Неверный режим отслеживания")
    for key, low, high, default in [("max_zoom", 1, 12, 6), ("eye_level", 0.1, 0.8, 0.38)]:
        profile.setdefault(key, default)
        value = profile[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Некорректное поле {key}")
    profile.setdefault("align_eyes", False)
    if type(profile["align_eyes"]) is not bool:
        raise ValueError("Некорректное поле align_eyes")
    profile.setdefault("motion_mode", "smooth")
    if profile["motion_mode"] not in ("smooth", "responsive"):
        raise ValueError("Неверный режим реакции")
    profile["subject_region"] = validate_region(profile.get("subject_region", [0, 0, 1, 1]))
    profile["framing"] = validate_framing(profile.get("framing"))
    calibration = profile.setdefault(
        "calibration", {"crop": [0, 0, 1, 1], "mirror": False, "zoom": 1, "offset_x": 0, "offset_y": 0}
    )
    if not isinstance(calibration, dict) or type(calibration.get("mirror")) is not bool:
        raise ValueError("Некорректная калибровка")
    calibration = dict(calibration)
    profile["calibration"] = calibration
    calibration.setdefault("rotation", 0)
    if type(calibration["rotation"]) is not int or calibration["rotation"] not in (0, 90, 180, 270):
        raise ValueError("Некорректный поворот камеры")
    for key, low, high in [("zoom", 0.5, 2), ("offset_x", -0.5, 0.5), ("offset_y", -0.5, 0.5)]:
        value = calibration.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Некорректная калибровка: " + key)
    crop = calibration.get("crop")
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in crop)
        or min(crop[:2]) < 0
        or min(crop[2:]) <= 0
        or crop[0] + crop[2] > 1 + 1e-9
        or crop[1] + crop[3] > 1 + 1e-9
    ):
        raise ValueError("Некорректная видимая область камеры")
    for key in ("roi", "target"):
        if profile.get(key) is not None:
            values = profile[key]
            if (
                not isinstance(values, list)
                or len(values) != 4
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
            ):
                raise ValueError(f"Некорректное поле {key}")
            rect = Rect(*values)
            if (
                rect.area <= 0
                or min(rect.x, rect.y) < 0
                or rect.x + rect.w > 1 + 1e-9
                or rect.y + rect.h > 1 + 1e-9
            ):
                raise ValueError(f"Область {key} должна находиться внутри изображения")
    profile.setdefault("remove_background", False)
    profile.setdefault("segmentation_quality", "balanced")
    profile.setdefault("background_color", "#000000")
    profile.setdefault("background_image", "")
    if type(profile["remove_background"]) is not bool or profile["segmentation_quality"] not in (
        "balanced",
        "accurate",
    ):
        raise ValueError("Неверные параметры удаления фона")
    if not isinstance(profile["background_color"], str) or not re.fullmatch(
        r"#[0-9a-fA-F]{6}", profile["background_color"]
    ):
        raise ValueError("Неверный цвет фона")
    if not isinstance(profile["background_image"], str):
        raise ValueError("Неверное изображение фона")
    slots = profile.setdefault("slots", [])
    if not isinstance(slots, list) or len(slots) > 4:
        raise ValueError("Поддерживается не больше четырех рамок")
    identifiers = set()
    clean_slots = []
    for raw in slots:
        if not isinstance(raw, dict):
            raise ValueError("Неверная рамка")
        slot = dict(raw)
        slot.setdefault("surface_lock", False)
        if type(slot["surface_lock"]) is not bool:
            raise ValueError("Неверная привязка рамки к поверхности")
        slot["frame_mask"] = FrameMask.from_record(slot.get("frame_mask")).record()
        slot.setdefault("framing_zoom", 1.0)
        if (
            type(slot["framing_zoom"]) not in (int, float)
            or not math.isfinite(slot["framing_zoom"])
            or not 0.5 <= slot["framing_zoom"] <= 2
        ):
            raise ValueError("Неверный масштаб рамки")
        slot["subject_region"] = validate_region(slot.get("subject_region", [0, 0, 1, 1]))
        slot["framing"] = validate_framing(slot.get("framing"))
        identifier = slot.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or len(identifier) > 80
            or identifier in identifiers
        ):
            raise ValueError("Идентификаторы рамок должны быть уникальными")
        identifiers.add(identifier)
        box = slot.get("target")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in box)
        ):
            raise ValueError("Неверные координаты рамки")
        if min(box[:2]) < 0 or min(box[2:]) <= 0 or box[0] + box[2] > 1 + 1e-9 or box[1] + box[3] > 1 + 1e-9:
            raise ValueError("Рамка выходит за пределы холста")
        if slot.get("shape") not in ("rectangle", "ellipse") or slot.get("subject_mode") not in (
            "face",
            "object",
            "upper",
            "person",
        ):
            raise ValueError("Неверный режим рамки")
        if type(slot.get("fill")) is not bool:
            raise ValueError("Неверное заполнение рамки")
        for key, low, high in [("padding", 0, 1), ("max_zoom", 1, 12)]:
            value = slot.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Неверные параметры рамки")
        eye = slot.get("eye_level")
        if eye is not None and (
            type(eye) not in (int, float) or not math.isfinite(eye) or not 0.1 <= eye <= 0.8
        ):
            raise ValueError("Неверный уровень глаз")
        clean_slots.append(slot)
    profile["slots"] = clean_slots
    profile.setdefault("active_slot", clean_slots[0]["id"] if clean_slots else "")
    if profile["active_slot"] not in identifiers and profile["active_slot"] != "":
        raise ValueError("Активная рамка отсутствует")
    return profile


def validate_region(values):
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
        or min(values[:2]) < 0
        or min(values[2:]) < 0.02
        or values[0] + values[2] > 1 + 1e-9
        or values[1] + values[3] > 1 + 1e-9
    ):
        raise ValueError("Выбранная часть должна находиться внутри цели и занимать не менее 2% её размера")
    return list(values)


def validate_framing(data):
    if data is None:
        return FramingRule().record()
    if not isinstance(data, dict):
        raise ValueError("Неверные правила автоматического кадрирования")
    value = FramingRule().record()
    value.update({key: data[key] for key in value if key in data})
    if type(value["enabled"]) is not bool or value["whole_when"] not in ("larger", "smaller"):
        raise ValueError("Неверное правило переключения части и целого")
    for key, low, high in [("threshold", 0.01, 0.95), ("hysteresis", 0, 0.3), ("transition", 0, 1)]:
        number = value[key]
        if type(number) not in (float, int) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError("Неверные параметры автоматического кадрирования: " + key)
    return value


def encode_selections(selections):
    return {
        key: {"kind": "quad", "value": [list(p) for p in value.quad]}
        if isinstance(value, QuadSelection)
        else {"kind": "box", "value": [value.x, value.y, value.w, value.h]}
        if isinstance(value, Rect)
        else {"kind": "point", "value": list(value)}
        for key, value in selections.items()
        if value is not None
    }


def decode_selections(data):
    if not isinstance(data, dict):
        raise ValueError("Неверные сохраненные цели")
    selections = {}
    for key, entry in data.items():
        if not isinstance(key, str) or key == "_reset" or not isinstance(entry, dict):
            raise ValueError("Неверная сохраненная цель")
        values = entry.get("value")
        if entry.get("kind") == "quad":
            if not valid_quad(values):
                raise ValueError("Неверная сохранённая поверхность")
            selections[key] = QuadSelection(tuple(tuple(float(v) for v in p) for p in values))
            continue
        count = 4 if entry.get("kind") == "box" else 2 if entry.get("kind") == "point" else 0
        if (
            not count
            or not isinstance(values, list)
            or len(values) != count
            or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
        ):
            raise ValueError("Неверные координаты сохраненной цели")
        if count == 4 and (
            min(values[2:]) <= 0 or values[0] + values[2] > 1 + 1e-9 or values[1] + values[3] > 1 + 1e-9
        ):
            raise ValueError("Неверная область сохраненной цели")
        selections[key] = Rect(*values) if count == 4 else tuple(values)
    return selections


class Preferences:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            if "profile" in data:
                validate_profile(data["profile"])
            clean_profiles = {}
            profiles = data.get("window_profiles", {})
            for key, entry in profiles.items() if isinstance(profiles, dict) else []:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    continue
                try:
                    clean_profiles[key] = {
                        "name": entry["name"],
                        "profile": validate_profile(entry.get("profile")),
                    }
                except (ValueError, TypeError):
                    continue
            data["window_profiles"] = clean_profiles
            clean_scenarios = {}
            scenarios = data.get("scenarios", {})
            for key, entry in scenarios.items() if isinstance(scenarios, dict) else []:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    continue
                try:
                    validate_profile(entry.get("profile"))
                    decode_selections(entry.get("selections", {}))
                    if entry.get("source") not in ("file", "photo", "camera", "screen") or not isinstance(
                        entry.get("path", ""), str
                    ):
                        continue
                    if entry.get("source") == "screen":
                        source = entry.get("screen_source")
                        if (
                            not isinstance(source, dict)
                            or source.get("kind") not in ("display", "window")
                            or type(source.get("id")) is not int
                            or source["id"] <= 0
                        ):
                            continue
                    if type(entry.get("position", 0)) is not int or entry.get("position", 0) < 0:
                        continue
                    if not isinstance(entry.get("camera_device", ""), str) or not isinstance(
                        entry.get("window", "demo"), str
                    ):
                        continue
                    clean_scenarios[key] = entry
                except (ValueError, TypeError):
                    continue
            data["scenarios"] = clean_scenarios
            if not isinstance(data.get("last_directory", ""), str):
                data["last_directory"] = ""
            return data
        except (OSError, ValueError, TypeError):
            return {}

    def save(self, data: dict, *, compact=False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".preferences-", suffix=".json", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    data,
                    stream,
                    ensure_ascii=False,
                    indent=None if compact else 2,
                    separators=(",", ":") if compact else None,
                    allow_nan=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(7):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError as error:
                    # Windows readers can briefly prevent replacing an existing file.
                    if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 6:
                        raise
                    time.sleep(min(0.01 * 2**attempt, 0.2))
        finally:
            temporary.unlink(missing_ok=True)
