"""User-requested local reproduction bundle: visible frames and non-secret settings."""

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def serial(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save_debug_bundle(main, directory):
    from . import __version__

    folder = Path(directory) / (time.strftime("FramePilot-case-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    for name, image in (
        ("input", main.input_view.image),
        ("target", main.target_view.image),
        ("output", main.last_output),
        ("interface", main.grab().toImage()),
    ):
        if not image.isNull() and not image.save(str(folder / (name + ".png"))):
            raise OSError("Не удалось сохранить кадр " + name)
    data = {
        "version": __version__,
        "source_kind": main.active_source,
        "selected_window": main.window_combo.currentText(),
        "profile": main.capture_profile(),
        "tracking": main.last_tracks,
        "signal": main.signal_report(),
        "frame_search": {
            "automatic": main.auto_slot.isChecked(),
            "manual": main.manual_slot,
            "candidates": [slot.rect for slot in main.slots],
        },
        "status": main.status.text(),
        "metrics": main.metrics.text(),
        "scope": "Local screenshots and settings selected by the user; original video is not copied",
    }
    (folder / "case.json").write_text(json.dumps(data, default=serial, ensure_ascii=False, indent=2))
    return folder
