# FramePilot third-party components

This local app bundles dynamically linked Qt frameworks and Python libraries.

- Python: Python Software Foundation License — https://docs.python.org/3/license.html
- PySide6 / Qt: LGPLv3, GPLv3 and applicable third-party licenses — https://www.qt.io/licensing/
  Qt libraries remain dynamically linked in the app bundle. Corresponding Qt/PySide sources are available at https://download.qt.io/official_releases/QtForPython/ and https://download.qt.io/official_releases/qt/.
- NumPy: BSD-3-Clause — https://numpy.org/doc/stable/license.html
- OpenCV: Apache-2.0 — https://opencv.org/license/
- Bundled YuNet 2023mar face model: MIT, Copyright (c) 2020 Shiqi Yu. Full license: framepilot/assets/YUNET_LICENSE.txt. Pinned source and SHA-256: framepilot/assets/yunet.json; https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet. No model download occurs at runtime.
- PyObjC: MIT — https://github.com/ronaldoussoren/pyobjc
- websocket-client: Apache-2.0 — https://github.com/websocket-client/websocket-client
- imageio-ffmpeg: BSD-2-Clause wrapper; bundled FFmpeg executable has its own build/license terms (including GPL-enabled libx264). Source/build information: https://github.com/imageio/imageio-ffmpeg and https://ffmpeg.org/download.html. Do not assume all FFmpeg copies in this bundle have the same license configuration.
- Qt Multimedia FFmpeg and OpenCV FFmpeg components: see their bundled license notices; FFmpeg source is available at https://ffmpeg.org/download.html.
- Included Eileen Collins test portrait: NASA public-domain image, via scikit-image. See framepilot/assets/README.md.

The app bundle includes dependency metadata and license files supplied by the build hooks. This is a local development build; review component-specific distribution obligations before publishing it.
