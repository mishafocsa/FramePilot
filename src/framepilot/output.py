"""Local-only MJPEG output for an OBS Browser Source, with no frame queue."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from . import __version__

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>FramePilot Output</title>
<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#000}
img{width:100%;height:100%;object-fit:contain}</style></head>
<body><img id="video" src="/stream.mjpg" alt=""><script>
const video=document.getElementById('video');
video.onerror=()=>setTimeout(()=>video.src='/stream.mjpg?t='+Date.now(),1000);
let instance=null;
setInterval(async()=>{try{const r=await fetch('/api/status',{cache:'no-store'});
const s=await r.json();if(instance&&instance!==s.instance)video.src='/stream.mjpg?t='+Date.now();
instance=s.instance;}catch(e){}},1500);
</script></body></html>"""


class LocalOutput:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.enabled = threading.Event()
        self.sequence = 0
        self.epoch = 0
        self.jpeg: bytes | None = None
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.width = self.height = 0
        self.frames = 0
        self.instance = ""
        self.hold_reason = None
        self.gate = 0
        self.health = {}

    def update_health(self, health):
        with self.condition:
            self.health = dict(health)

    def status(self):
        with self.condition:
            return {
                "application": "FramePilot",
                "version": __version__,
                "instance": self.instance,
                "width": self.width,
                "height": self.height,
                "frames": self.frames,
                "hold_reason": self.hold_reason,
                "signal": dict(self.health),
            }

    def hold(self, image, reason):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Could not encode fallback frame")
        with self.condition:
            self.gate += 1
            self.hold_reason = reason
            self.jpeg = encoded.tobytes()
            self.height, self.width = image.shape[:2]
            self.sequence += 1
            self.condition.notify_all()

    def release(self):
        with self.condition:
            self.gate += 1
            self.hold_reason = None

    def reset(self, epoch: int, clear: bool = True) -> None:
        if not clear:
            with self.condition:
                self.epoch = epoch
            return
        ok, black = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
        with self.condition:
            self.epoch = epoch
            if self.hold_reason is None:
                self.jpeg = black.tobytes() if ok else None
            self.frames = 0
            self.sequence += 1
            self.condition.notify_all()

    def publish(self, image: np.ndarray, epoch: int | None = None, deadline: float | None = None) -> None:
        if not self.enabled.is_set():
            return
        with self.condition:
            if self.hold_reason is not None:
                return
            gate = self.gate
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return
        with self.condition:
            if epoch is not None and self.epoch != epoch:
                return
            if self.hold_reason is not None or self.gate != gate:
                return
            if deadline is not None and time.monotonic() > deadline:
                return
            if not self.enabled.is_set():
                return
            self.jpeg = encoded.tobytes()
            self.height, self.width = image.shape[:2]
            self.frames += 1
            self.sequence += 1
            self.condition.notify_all()

    @property
    def url(self) -> str | None:
        return f"http://127.0.0.1:{self.server.server_port}/framepilot" if self.server else None

    def start(self, port: int = 18765) -> str:
        if self.server:
            return self.url
        publisher = self
        stopped = threading.Event()
        self._stopped = stopped
        self.instance = str(time.time_ns())

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/framepilot"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(PAGE)
                    return
                if path == "/api/status":
                    status = publisher.status()
                    data = json.dumps(status).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.connection.settimeout(3)
                last = -1
                try:
                    self.wfile.write(b"--frame\r\n")
                    while not stopped.is_set():
                        with publisher.condition:
                            publisher.condition.wait_for(
                                lambda: (
                                    (publisher.jpeg is not None and publisher.sequence != last)
                                    or stopped.is_set()
                                ),
                                0.25,  # Keep Chromium's decoder live for photos and paused video.
                            )
                            if stopped.is_set():
                                break
                            frame, sequence = publisher.jpeg, publisher.sequence
                        if frame is None:
                            continue
                        last = sequence
                        # Complete this part now, without waiting for a later image.
                        # Reuse the JPEG on idle ticks; processing/FPS counters stay unchanged.
                        self.wfile.write(
                            b"Content-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode()
                            + b"\r\n\r\n"
                            + frame
                            + b"\r\n--frame\r\n"
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    pass

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError:
            if port == 0:
                raise
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.enabled.set()
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        self.thread.start()
        return self.url

    def stop(self) -> None:
        self.enabled.clear()
        with self.condition:
            self.gate += 1
        if hasattr(self, "_stopped"):
            self._stopped.set()
        with self.condition:
            self.condition.notify_all()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        if self.hold_reason is None:
            self.jpeg = None
