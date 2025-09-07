from __future__ import annotations
import json
from typing import Dict, Any
import zmq


class SensorBus:
    """
    Very small ZMQ SUB that accepts JSON blobs.
    Expected payload: {"topic": "...", "data": {...}}
    Caches last value per topic in .latest.
    """

    def __init__(self, endpoint: str = "tcp://localhost:5556"):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(endpoint)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.setsockopt(zmq.RCVTIMEO, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.latest: Dict[str, Any] = {}

    def poll(self, max_msgs: int = 10) -> int:
        count = 0
        for _ in range(max_msgs):
            try:
                msg = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                obj = json.loads(msg.decode("utf-8"))
                topic = obj.get("topic", "unknown")
                data = obj.get("data", obj)
                self.latest[topic] = data
                count += 1
            except Exception:
                pass
        return count

    def get(self, topic: str, default=None):
        return self.latest.get(topic, default)

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:
            pass
