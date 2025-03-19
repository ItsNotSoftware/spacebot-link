import sys
import signal
import zmq
import cv2
import numpy as np
import pyzed.sl as sl
import time
from collections import deque

exit_app = False

def signal_handler(sig, frame):
    global exit_app
    exit_app = True

def log_message(message):
    print(f"[ZMQ Stream] {message}")

def open_zed_camera(resolution=sl.RESOLUTION.HD720):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.sdk_verbose = 1
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        log_message("Failed to open ZED camera")
        return None
    return zed

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <delay_ms>")
        sys.exit(1)

    delay_ms = int(sys.argv[1])
    frame_queue = deque()

    zed = open_zed_camera()
    if not zed:
        sys.exit(1)

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    signal.signal(signal.SIGINT, signal_handler)
    
    left_image = sl.Mat()
    log_message(f"ZMQ streaming started on port 5555 with {delay_ms} ms delay")
    
    while not exit_app:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            left_cv = left_image.get_data()[:, :, :3]  # Extract RGB
            _, buffer = cv2.imencode(".jpg", left_cv)
            
            timestamp = time.time()
            frame_queue.append((buffer, timestamp))

        # Send delayed frames
        while frame_queue:
            front = frame_queue[0]
            elapsed = (time.time() - front[1]) * 1000  # Convert to ms
            
            if elapsed >= delay_ms:
                publisher.send(front[0].tobytes())
                frame_queue.popleft()
            else:
                break
        
        time.sleep(1/30)  # 30 FPS

    log_message("Shutting down...")
    zed.close()
    publisher.close()
    context.term()

if __name__ == "__main__":
    main()
