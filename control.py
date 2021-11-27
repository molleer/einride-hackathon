from dataclasses import dataclass

import cv2
import numpy as np
from numpy.lib.function_base import average

from utils import CannyEdge, region_of_interest, convergence, average_slope_intercept

rho = 2
theta = np.pi / 180
threshold = 20

alpha = 10
speed = 0.2


@dataclass
class DriveCommand:
    angle: float
    throttle: float


def FrameToCommand(frame):
    edge_image = CannyEdge(frame)
    cropped_image = region_of_interest(edge_image)
    lines = cv2.HoughLinesP(
        cropped_image,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=10,
        maxLineGap=30,
    )
    averaged_kms = average_slope_intercept(lines)
    (x, y) = convergence(averaged_kms[0], averaged_kms[1])

    print((x, y))
    steering = alpha * (x - 80) / 80
    steering = min(1.0, steering)
    steering = max(-1.0, steering)

    rtn = DriveCommand(steering, speed)
    print(rtn)

    return rtn
