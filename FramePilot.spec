# Build with: .venv/bin/python -m PyInstaller --noconfirm FramePilot.spec
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH)
datas = collect_data_files("framepilot", includes=["assets/*"])
datas += [(str(root / "packaging" / "THIRD_PARTY_NOTICES.md"), ".")]
packages = ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6", "numpy",
            "opencv-python-headless", "websocket-client", "imageio-ffmpeg"]
native_imports = []
if sys.platform == "darwin":
    packages += ["pyobjc-core", "pyobjc-framework-Cocoa",
                "pyobjc-framework-Quartz", "pyobjc-framework-Vision", "pyobjc-framework-ScreenCaptureKit",
                "pyobjc-framework-CoreMedia", "pyobjc-framework-CoreML"]
    native_imports = ["Vision", "ScreenCaptureKit", "Quartz", "Foundation", "CoreMedia", "CoreML", "objc"]
for package in packages:
    datas += copy_metadata(package)

analysis = Analysis(
    [str(root / "packaging" / "entrypoint.py")],
    pathex=[str(root / "src")],
    datas=datas,
    hiddenimports=native_imports,
    excludes=["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtPdf", "pytest", "ruff"],
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="FramePilot", console=False,
          argv_emulation=False, target_arch="arm64" if sys.platform == "darwin" else None, codesign_identity=None)
collection = COLLECT(exe, analysis.binaries, analysis.datas, name="FramePilot")
if sys.platform == "darwin":
    app = BUNDLE(collection, name="FramePilot.app", bundle_identifier="local.framepilot.desktop",
             icon=str(root / "src" / "framepilot" / "assets" / "FramePilot.icns"),
             info_plist={"CFBundleShortVersionString": "0.15.4", "CFBundleVersion": "19",
                         "LSMinimumSystemVersion": "14.0", "NSHighResolutionCapable": True,
                         "NSPrincipalClass": "NSApplication", "NSRequiresAquaSystemAppearance": False,
                         "NSScreenCaptureUsageDescription": "FramePilot захватывает выбранный экран или окно для локальной обработки видео и поиска рамки.",
                         "NSCameraUsageDescription": "FramePilot обрабатывает изображение выбранной камеры локально и передаёт его в ваш видеовыход."})
