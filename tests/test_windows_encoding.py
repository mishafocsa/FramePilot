import json
import os
import subprocess
import sys


def test_diagnostic_report_roundtrips_unicode_with_non_utf8_default_locale(tmp_path):
    script = r"""
import json, locale, sys
from pathlib import Path
from types import SimpleNamespace as NS
from framepilot.debug_bundle import save_debug_bundle

class EmptyImage:
    def isNull(self):
        return True

image = EmptyImage()
label = "\u0422\u0435\u0441\u0442\u043e\u0432\u043e\u0435 \u043e\u043a\u043d\u043e \U0001f34e"
main = NS(
    input_view=NS(image=image), target_view=NS(image=image), last_output=image,
    grab=lambda: NS(toImage=lambda: image), active_source="photo",
    window_combo=NS(currentText=lambda: label), capture_profile=lambda: {"label": label},
    last_tracks={}, signal_report=lambda: {}, auto_slot=NS(isChecked=lambda: False),
    manual_slot=False, slots=[], status=NS(text=lambda: label), metrics=NS(text=lambda: label),
)
folder = save_debug_bundle(main, Path(sys.argv[1]))
data = json.loads((folder / "case.json").read_bytes().decode("utf-8"))
assert data["selected_window"] == label
assert data["profile"]["label"] == label
assert data["status"] == label
print(json.dumps({"passed": True, "default_encoding": locale.getencoding()}))
"""
    environment = dict(os.environ, PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert json.loads(result.stdout)["passed"]
