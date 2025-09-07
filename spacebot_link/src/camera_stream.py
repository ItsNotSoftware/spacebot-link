# camera_stream.py
from __future__ import annotations
import cv2
import numpy as np
import zmq
from typing import Optional, Tuple


class CameraStream:
    """
    ZMQ SUB that receives JPEG frames (raw bytes).
    Decodes to RGB; latest frame available via .frame_rgb.
    """

    def __init__(self, endpoint: str = "tcp://localhost:5555"):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(endpoint)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.setsockopt(zmq.RCVTIMEO, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._last_rgb: Optional[np.ndarray] = None
        self._w, self._h = 1280, 720  # default until first frame

    @property
    def size(self) -> Tuple[int, int]:
        return (self._w, self._h)

    @property
    def aspect(self) -> float:
        return self._h / self._w if self._w else 9 / 16

    @property
    def frame_rgb(self) -> Optional[np.ndarray]:
        return self._last_rgb

    def poll(self) -> bool:
        """Try to receive and decode one frame; return True if updated."""
        try:
            msg = self._sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return False

        arr = np.frombuffer(msg, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False

        # keep your original vertical flip
        img_bgr = np.flipud(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._h, self._w = img_rgb.shape[:2]
        self._last_rgb = img_rgb
        return True

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:
            pass
