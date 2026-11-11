# Test portrait

`astronaut.png` is the Eileen Collins portrait distributed by scikit-image, originally from NASA.

- Source: https://github.com/scikit-image/scikit-image/blob/v0.14.2/skimage/data/astronaut.png
- Dataset documentation and public-domain status: https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut

This fixture is used offline to test actual Apple Vision detection and moving-video playback. It is not a face identity model.

The SVG control glyphs and FramePilot application icon are original project assets. The application icon is built from the same vector brand mark used by the editor; regenerate its PNG and ICNS with `python tools/build_icon.py` on macOS.
# YuNet

`face_detection_yunet_2023mar.onnx` is bundled from OpenCV Zoo for local inference with OpenCV 4.x. See `yunet.json` for the pinned repository revision, source URL and SHA-256. MIT license and copyright are reproduced in `YUNET_LICENSE.txt`. No runtime download is performed. This is a face detector, not a face-identity model or arbitrary-object detector.

