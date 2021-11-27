
import websocket
import _thread
import time
import cv2
import numpy as np
from time import sleep

from control import FrameToCommand, DriveCommand

host = "192.168.171.132"
port = 8887

socket_address = f"ws://{host}:{port}/wsDrive"
video_address = f"http://{host}:{port}/video"


def on_message(ws, message):
    # print(message)
    pass


def on_error(ws, error):
    print(error)
    message = f"{{\"angle\":{0.0},\"throttle\":{0.0},\"drive_mode\":\"user\",\"recording\":false}}"
    ws.send(message)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")
    message = f"{{\"angle\":{0.0},\"throttle\":{0.0},\"drive_mode\":\"user\",\"recording\":false}}"
    ws.send(message)


def on_open(ws):
    def run(*args):
        # your car logic here

        cap = cv2.VideoCapture(video_address)

        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]
        steering_list = []

        while True:
            ret, frame = cap.read()

            # do something based on the frame
            if ret:
                command = FrameToCommand(frame)
            else:
                command = DriveCommand(0, 0)
            # print(f"command: {command}")

            angle = 0.0
            throttle = 0.2

            steering_list.append(command.angle)
            if len(steering_list) >= 2:
                steering_list.pop(0)

            avg_angle = np.average(steering_list)

            message = f"{{\"angle\":{avg_angle},\"throttle\":{command.throttle},\"drive_mode\":\"user\",\"recording\":false}}"
            ws.send(message)
            sleep(0.2)
            message = f"{{\"angle\":{0.0},\"throttle\":{0.0},\"drive_mode\":\"user\",\"recording\":false}}"
            ws.send(message)
            sleep(0.1)
            # print(message)


    _thread.start_new_thread(run, ())




if __name__ == "__main__":
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp(socket_address,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()
