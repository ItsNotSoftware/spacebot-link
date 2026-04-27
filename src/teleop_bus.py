from __future__ import annotations
import base64
import binascii
import json
from typing import Any, Dict, Optional

import cv2
import numpy as np
import zmq


class TeleopBusSub:
    """
    One ZMQ SUB socket that receives JSON blobs of the form:
        {"topic": "/some/topic", "data": {...}}

    It maintains a cache (last value per topic) and provides helpers to decode
    `sensor_msgs/Image`-like payloads where `data` is base64-encoded.
    """

    def __init__(
        self,
        endpoint: str = "tcp://localhost:5556",
        subscribe_prefix: Optional[bytes] = None,
        rcv_timeout_ms: int = 1,
        rcv_hwm: int = 200,
        conflate: bool = False,
    ) -> None:
        """Open a SUB socket against `endpoint` with the given filter and queue settings."""
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(endpoint)
        self._sock.setsockopt(zmq.RCVTIMEO, rcv_timeout_ms)
        self._sock.setsockopt(zmq.RCVHWM, rcv_hwm)
        self._sock.setsockopt(zmq.LINGER, 0)
        if conflate:
            self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, subscribe_prefix or b"")
        self.latest: Dict[str, Any] = {}

    # ---------- polling & cache ----------
    def poll(self, max_msgs: int = 1000) -> int:
        """Pump up to `max_msgs` messages from the socket into `self.latest`.
        Returns the number of processed messages.
        """
        count = 0
        for _ in range(max_msgs):
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                obj = json.loads(raw.decode("utf-8"))
                topic = obj.get("topic", "")
                data = obj.get("data", obj)
                if isinstance(topic, str) and topic:
                    self.latest[topic] = data
                    count += 1
            except Exception:
                pass
        return count

    def get(self, topic: str, default=None) -> Any:
        """Return cached payload for topic or default."""
        return self.latest.get(topic, default)

    # ---------- image helpers ----------
    def get_image_rgb(self, topic: str) -> Optional[np.ndarray]:
        """Return latest image for topic as an RGB numpy array, or None."""
        payload = self.get(topic)
        if not isinstance(payload, dict):
            return None
        return self._decode_image_message(payload)

    def _decode_image_message(self, payload: Dict[str, Any]) -> Optional[np.ndarray]:
        """Decode ROS-like image dict (base64 data + metadata) into RGB array."""
        try:
            width = int(payload.get("width"))  # type: ignore
            height = int(payload.get("height"))  # type: ignore
            encoding = str(payload.get("encoding", "")).lower()
            data64 = payload.get("data")
            if not isinstance(data64, str) or width <= 0 or height <= 0:
                return None
            buf = base64.b64decode(data64)
        except (TypeError, ValueError, binascii.Error):
            return None

        arr = np.frombuffer(buf, dtype=np.uint8)

        if encoding in ("rgb8", "bgr8", "rgba8", "bgra8"):
            channels = 4 if "a" in encoding else 3
            expected = width * height * channels
            if arr.size < expected:
                return None
            frame = arr[:expected].reshape((height, width, channels))
            if encoding == "rgb8":
                rgb = frame[..., :3]
            elif encoding == "bgr8":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif encoding == "rgba8":
                rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            else:  # bgra8
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            return np.ascontiguousarray(rgb)

        if encoding in ("mono8", "8uc1"):
            expected = width * height
            if arr.size < expected:
                return None
            mono = arr[:expected].reshape((height, width))
            rgb = cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
            return np.ascontiguousarray(rgb)

        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        """Close subscriber socket."""
        try:
            self._sock.close(0)
        except Exception:
            pass


class TeleopBusPub:
    """
    Minimal ZMQ PUB socket that sends JSON blobs:
        {"topic": "/topic", "data": {...}}

    Intended for UI -> robot command messages (e.g., /space_cobot/cmd_vel).
    """

    def __init__(
        self,
        endpoint: str = "tcp://localhost:5557",
        snd_hwm: int = 1000,
    ) -> None:
        """Open a PUB socket against `endpoint` with the given send queue size."""
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, snd_hwm)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)

    def publish(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish JSON payload on a topic via PUB socket."""
        try:
            msg = json.dumps({"topic": topic, "data": data})
            self._sock.send_string(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception:
            pass

    def close(self) -> None:
        """Close publisher socket."""
        try:
            self._sock.close(0)
        except Exception:
            pass
