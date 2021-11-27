import cv2
import numpy as np
from numpy.lib.function_base import average

from utils import CannyEdge, region_of_interest, convergence, average_slope_intercept


def display_lines(image, lines, color=(255, 0, 0)):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), color, 2)
    return line_image

def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*1/2)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [x1, y1, x2, y2]

def fits_to_points(image, fits):
    left_line = make_points(image, fits[0])
    right_line = make_points(image, fits[1])
    return np.array((left_line, right_line))

video_address = "http://192.168.171.132:8887/video"

cap = cv2.VideoCapture(video_address)

ret, frame = cap.read()

line_image = CannyEdge(frame)
cv2.imwrite("line_image.png", line_image)
cv2.imwrite("mask_image.png", region_of_interest(line_image))


video_address = "http://192.168.171.132:8887/video"

cap = cv2.VideoCapture(video_address)

ret, frame = cap.read()
cv2.imwrite("frame.png", frame)

edge_image = CannyEdge(frame)
cv2.imwrite("line_image.png", edge_image)


cropped_image = region_of_interest(edge_image)
cv2.imwrite("mask_image.png", cropped_image)

rho = 2
theta = np.pi / 180
threshold = 20

lines = cv2.HoughLinesP(
    cropped_image, rho, theta, threshold, np.array([]), minLineLength=30, maxLineGap=2
)
line_image = display_lines(frame, lines)
cv2.imwrite("line_image.png", line_image)

combo_image = cv2.addWeighted(frame, 0.8, line_image,  1,1)
cv2.imwrite("combo_image.png", combo_image)

averaged_kms = average_slope_intercept(lines)
averaged_points = fits_to_points(frame, averaged_kms)

averaged_image = display_lines(frame, averaged_points)
cv2.imwrite("averaged_image.png", averaged_image)
print("Molle why?")
print(averaged_points[0])
print(averaged_points[1])
print(averaged_kms[0], averaged_kms[1])
(x, y) = convergence(averaged_kms[0], averaged_kms[1])
print((x, y))

move_vector = np.array(
        [
            [
                (60, 120),
                (x, y),
            ]
        ],
        np.int32,
    )


averaged_combo_image = cv2.addWeighted(frame, 0.8, averaged_image, 1, 1)
cv2.imwrite("averaged_combo_image.png", averaged_combo_image)

direction_image = display_lines(frame, move_vector, color=(0, 255, 0))
cv2.imwrite("direction_image.png", direction_image)
direction_combo_image = cv2.addWeighted(averaged_combo_image, 0.8, direction_image, 1, 1)
cv2.imwrite("direction_combo_image.png", direction_combo_image)


if __name__ == "__main__":
    while True:
        ret, frame = cap.read()

        edge_image = CannyEdge(frame)


        cropped_image = region_of_interest(edge_image)



        lines = cv2.HoughLinesP(
            cropped_image, rho, theta, threshold, np.array([]), minLineLength=10, maxLineGap=30
        )
        line_image = display_lines(frame, lines)

        combo_image = cv2.addWeighted(frame, 0.8, line_image,  1,1)

        averaged_kms = average_slope_intercept(lines)
        averaged_points = fits_to_points(frame, averaged_kms)
        (x, y) = convergence(averaged_kms[0], averaged_kms[1])

        move_vector = np.array(
                [
                    [
                        (80, 120),
                        (x, y),
                    ]
                ],
                np.int32,
            )

        averaged_image = display_lines(frame, averaged_points)
        averaged_combo_image = cv2.addWeighted(frame, 0.8, averaged_image, 1, 1)

        direction_image = display_lines(frame, move_vector, color=(0, 255, 0))
        direction_combo_image = cv2.addWeighted(averaged_combo_image, 0.8, direction_image, 1, 1)
        cv2.imwrite("direction_combo_image.png", direction_combo_image)

