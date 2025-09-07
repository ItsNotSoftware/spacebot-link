#!/usr/bin/env python3
import sys
import signal
import time
from collections import deque
from typing import Deque, Optional, Tuple, Union

import cv2
import numpy as np
import zmq

exit_app: bool = False


def signal_handler(sig: int, frame) -> None:
    """Handle Ctrl+C to exit gracefully."""
    global exit_app
    exit_app = True


def log_message(msg: str) -> None:
    """Print a message with a ZMQ prefix."""
    print(f"[ZMQ Stream] {msg}", flush=True)


# --------------------- OpenCV webcam ---------------------


def open_cv_camera(
    device_index: int = 0,
) -> Optional[Tuple[cv2.VideoCapture, int, int, float]]:
    """
    Open a webcam using OpenCV without overriding default resolution.
    Returns (cap, width, height, fps) or None if failed.
    """
    cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)
    if not cap.isOpened():
        return None

    actual_w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps: float = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if actual_fps <= 1e-3:
        actual_fps = 30.0

    return cap, actual_w, actual_h, actual_fps


# ----------------------- ZED camera ----------------------


def open_zed_camera() -> (
    Optional[Tuple["sl.Camera", "sl.Mat", float]]
):  # forward-decl types
    """
    Open a ZED camera using pyzed.sl.
    Returns (zed, left_image, fps) or None if failed.
    """
    try:
        import pyzed.sl as sl
    except Exception:
        log_message(
            "pyzed.sl not available. Install ZED SDK Python API to use 'zed' mode."
        )
        return None

    zed: "sl.Camera" = sl.Camera()
    init_params: "sl.InitParameters" = sl.InitParameters()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        log_message(f"Failed to open ZED camera: {err}")
        return None

    left_image: "sl.Mat" = sl.Mat()
    fps: float = float(zed.get_camera_information().camera_configuration.fps or 30)
    return zed, left_image, fps


# --------------------------- Main ------------------------


def main() -> None:
    """
    Usage:
        python stream_zmq.py <cv|zed> <delay_ms> [device_index]

    Examples:
        python stream_zmq.py cv 0
        python stream_zmq.py cv 150 1
        python stream_zmq.py zed 200
    """
    if len(sys.argv) < 3:
        print(main.__doc__)
        sys.exit(1)

    mode: str = sys.argv[1].lower()
    delay_ms: int = int(sys.argv[2])
    device_index: int = int(sys.argv[3]) if len(sys.argv) >= 4 else 0

    # ---- Open source ----
    if mode == "cv":
        opened = open_cv_camera(device_index)
        if opened is None:
            log_message(f"Failed to open webcam (device {device_index})")
            sys.exit(1)
        cap, actual_w, actual_h, actual_fps = opened
        source_desc: str = (
            f"CV dev={device_index} {actual_w}x{actual_h}@{actual_fps:.1f}"
        )
        is_zed: bool = False
    elif mode == "zed":
        opened = open_zed_camera()
        if opened is None:
            sys.exit(1)
        zed, left_image, actual_fps = opened
        source_desc = f"ZED @{actual_fps:.1f} FPS (default resolution)"
        is_zed = True
    else:
        log_message("First argument must be 'cv' or 'zed'")
        sys.exit(1)

    # ---- ZMQ PUB ----
    context: zmq.Context = zmq.Context.instance()
    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    signal.signal(signal.SIGINT, signal_handler)

    frame_queue: Deque[Tuple[np.ndarray, float]] = deque()
    log_message(f"Streaming {source_desc} over tcp://*:5555 with {delay_ms} ms delay")

    target_dt: float = 1.0 / (actual_fps if actual_fps > 0 else 30.0)

    # ---- Loop ----
    while not exit_app:
        if is_zed:
            import pyzed.sl as sl  # local import to keep annotations above working

            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                frame: np.ndarray = left_image.get_data()[:, :, :3]  # RGB
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    frame_queue.append((buf, time.time()))
        else:
            ok, frame_bgr = cap.read()
            if ok:
                ok2, buf = cv2.imencode(".jpg", frame_bgr)
                if ok2:
                    frame_queue.append((buf, time.time()))
            else:
                time.sleep(0.01)

        # Send delayed frames
        now: float = time.time()
        while frame_queue:
            buf, ts = frame_queue[0]
            if (now - ts) * 1000.0 >= delay_ms:
                publisher.send(buf.tobytes())
                frame_queue.popleft()
            else:
                break

        # Pace loop
        time.sleep(max(0.0, target_dt - (time.time() - now)))

    # ---- Cleanup ----
    log_message("Shutting down…")
    try:
        if is_zed:
            zed.close()
        else:
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
