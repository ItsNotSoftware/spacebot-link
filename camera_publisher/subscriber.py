import sys
import zmq
import cv2
import numpy as np

def log_message(message):
    print(f"[ZMQ Receiver] {message}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <zmq_address>")
        sys.exit(1)

    zmq_address = sys.argv[1]
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(zmq_address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    log_message(f"Subscribed to {zmq_address}")
    
    while True:
        try:
            message = subscriber.recv()
            np_arr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                cv2.imshow("ZMQ Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            log_message("Shutting down...")
            break

    subscriber.close()
    context.term()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()