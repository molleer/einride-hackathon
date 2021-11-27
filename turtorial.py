from dataclasses import dataclass

import websocket
import _thread
import time
import cv2
import numpy as np

host = "192.168.171.132"
port = 8887

socket_address = f"ws://{host}:{port}/wsDrive"
video_address = f"http://{host}:{port}/video"

@dataclass
class DriveCommand:
    angle: float
    throttle: float

def FrameToCommand(ret, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rtn = DriveCommand(0.0, 0.2)

    return rtn

def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    def run(*args):
        # your car logic here

        cap = cv2.VideoCapture(video_address)

        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,50,150,apertureSize = 3)
            lines = cv2.HoughLines(edges,1,np.pi/180, 200)

            angle = 0.0
            throttle = 0.2

            message = f"{{\"angle\":{command.angle},\"throttle\":{command.throttle},\"drive_mode\":\"user\",\"recording\":false}}"
            ws.send(message)
            # print(message)

    _thread.start_new_thread(run, ())


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(socket_address,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()
