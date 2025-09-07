#!/usr/bin/env python3
import sys
import signal
import time
from collections import deque

import cv2
import numpy as np
import zmq

exit_app = False


def signal_handler(sig, frame):
    global exit_app
    exit_app = True


def log_message(msg: str):
    print(f"[ZMQ Stream] {msg}", flush=True)


def open_webcam(device_index=0, width=1280, height=720, fps=30):
    cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)
    if not cap.isOpened():
        return None

    # Try to set desired properties (may be ignored by some drivers)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    # Read back what we actually got
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    # Some cameras report 0 FPS—fallback to requested
    actual_fps = float(actual_fps) if actual_fps and actual_fps > 1e-3 else float(fps)

    return cap, actual_w, actual_h, actual_fps


def main():
    """
    Usage:
        python stream_webcam_zmq.py <delay_ms> [device_index] [width height fps]

    Examples:
        python stream_webcam_zmq.py 0
        python stream_webcam_zmq.py 150 1 1280 720 30
    """
    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit(1)

    delay_ms = int(sys.argv[1])
    device_index = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    width = int(sys.argv[3]) if len(sys.argv) >= 4 else 1280
    height = int(sys.argv[4]) if len(sys.argv) >= 5 else 720
    fps = int(sys.argv[5]) if len(sys.argv) >= 6 else 30

    opened = open_webcam(device_index, width, height, fps)
    if opened is None:
        log_message(f"Failed to open webcam (device {device_index})")
        sys.exit(1)
    cap, actual_w, actual_h, actual_fps = opened

    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    signal.signal(signal.SIGINT, signal_handler)

    frame_queue: deque[tuple[np.ndarray, float]] = deque()
    log_message(
        f"Streaming webcam dev={device_index} at {actual_w}x{actual_h}@{actual_fps:.1f} "
        f"over tcp://*:5555 with {delay_ms} ms delay"
    )

    # Target loop timing (don’t spam CPU if camera is faster/weird FPS)
    target_dt = 1.0 / (actual_fps if actual_fps > 0 else fps)

    while not exit_app:
        ok, frame_bgr = cap.read()
        if ok:
            # Encode to JPEG (BGR is fine — your client already converts BGR->RGB+flip)
            # Tune quality if you want: (cv2.IMWRITE_JPEG_QUALITY, 85)
            success, buf = cv2.imencode(".jpg", frame_bgr)
            if success:
                frame_queue.append((buf, time.time()))
        else:
            # Small nap if camera hiccups
            time.sleep(0.01)

        # Send delayed frames in order
        now = time.time()
        while frame_queue:
            buf, ts = frame_queue[0]
            elapsed_ms = (now - ts) * 1000.0
            if elapsed_ms >= delay_ms:
                publisher.send(buf.tobytes())
                frame_queue.popleft()
            else:
                break

        # Pace the loop roughly to camera FPS
        time.sleep(max(0.0, target_dt - (time.time() - now)))

    log_message("Shutting down…")
    try:
        cap.release()
    except Exception:
        pass
    try:
        publisher.close(0)
        context.term()
    except Exception:
        pass


if __name__ == "__main__":
    main()
