"""On-device human segmentation. Its mask includes all visible people, not arbitrary objects."""

from pathlib import Path

import cv2
import numpy as np


def vision_image(bgr, max_side=640):
    import Quartz as q

    h, w = bgr.shape[:2]
    scale = min(1, max_side / max(w, h))
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale)))), cv2.COLOR_BGR2RGB
    )
    sh, sw = rgb.shape[:2]
    data = rgb.tobytes()
    provider = q.CGDataProviderCreateWithData(None, data, len(data), None)
    image = q.CGImageCreate(
        sw,
        sh,
        8,
        24,
        sw * 3,
        q.CGColorSpaceCreateDeviceRGB(),
        q.kCGImageAlphaNone,
        provider,
        None,
        False,
        q.kCGRenderingIntentDefault,
    )
    return image, data


class PersonSegmenter:
    def __init__(self):
        import Quartz
        import Vision

        self.q, self.v = Quartz, Vision
        self.requests = {}

    def mask(self, image, quality="balanced"):
        q, v = self.q, self.v
        if quality not in self.requests:
            request = v.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
            request.setQualityLevel_(
                v.VNGeneratePersonSegmentationRequestQualityLevelAccurate
                if quality == "accurate"
                else v.VNGeneratePersonSegmentationRequestQualityLevelBalanced
            )
            request.setOutputPixelFormat_(q.kCVPixelFormatType_OneComponent8)
            self.requests[quality] = request
        request = self.requests[quality]
        cg, backing = vision_image(image)
        handler = v.VNImageRequestHandler.alloc().initWithCGImage_orientation_options_(cg, 1, None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Сегментация человека: {error}")
        observation = next(iter(request.results() or []), None)
        if observation is None:
            return np.zeros(image.shape[:2], np.float32)
        pixel = observation.pixelBuffer()
        flags = q.kCVPixelBufferLock_ReadOnly
        q.CVPixelBufferLockBaseAddress(pixel, flags)
        try:
            width, height = q.CVPixelBufferGetWidth(pixel), q.CVPixelBufferGetHeight(pixel)
            stride = q.CVPixelBufferGetBytesPerRow(pixel)
            address = q.CVPixelBufferGetBaseAddress(pixel)
            buffer = address.as_buffer(height * stride) if hasattr(address, "as_buffer") else address
            mask = (
                np.frombuffer(buffer, np.uint8, count=height * stride)
                .reshape(height, stride)[:, :width]
                .copy()
            )
        finally:
            q.CVPixelBufferUnlockBaseAddress(pixel, flags)
        return cv2.resize(mask.astype(np.float32) / 255, (image.shape[1], image.shape[0]))


class Background:
    def __init__(self):
        self.key = None
        self.image = None

    def frame(self, settings):
        color = settings.background_color.lstrip("#")
        rgb = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
        canvas = np.full((settings.height, settings.width, 3), rgb[::-1], np.uint8)
        if not settings.background_image:
            return canvas
        path = Path(settings.background_image)
        key = str(path), path.stat().st_mtime_ns, settings.width, settings.height
        if key != self.key:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Не удалось прочитать фон. Выберите PNG/JPEG или сбросьте изображение фона.")
            h, w = image.shape[:2]
            scale = max(settings.width / w, settings.height / h)
            image = cv2.resize(
                image, (max(settings.width, round(w * scale)), max(settings.height, round(h * scale)))
            )
            x, y = (image.shape[1] - settings.width) // 2, (image.shape[0] - settings.height) // 2
            self.image = image[y : y + settings.height, x : x + settings.width].copy()
            if self.image.ndim == 2:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.key = key
        if self.image.ndim == 3 and self.image.shape[2] == 4:
            return composite(self.image[:, :, :3], self.image[:, :, 3].astype(np.float32) / 255, canvas)
        return self.image.copy()


def composite(layer, alpha, background):
    # Blending a small slot must not allocate several full-HD float canvases.
    result = background.copy()
    x, y, width, height = cv2.boundingRect((alpha > 0).astype(np.uint8))
    if width == 0 or height == 0:
        return result
    source = layer[y : y + height, x : x + width]
    destination = result[y : y + height, x : x + width]
    opacity = np.clip(alpha[y : y + height, x : x + width], 0, 1)[..., None]
    if np.all(opacity == 1):
        destination[:] = source
    else:
        destination[:] = np.clip(
            source.astype(np.float32) * opacity + destination.astype(np.float32) * (1 - opacity), 0, 255
        ).astype(np.uint8)
    return result
