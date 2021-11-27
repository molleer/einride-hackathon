import cv2
import numpy as np
from numpy.lib.function_base import average


def CannyEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cannyImage = cv2.Canny(blur, 60, 180)
    return cannyImage


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    triangle = np.array(
        [
            [
                (0, 80),
                (width // 2, height // 2),
                (width, 80),
                (width, height),
                (0, height),
            ]
        ],
        np.int32,
    )
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines, color=(255, 0, 0)):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), color, 2)
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_points(image, left_fit_average)[0]
    right_line = make_points(image, right_fit_average)[0]
    return (np.array((left_line, right_line)), [left_fit_average, right_fit_average])

def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*1/2)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def convergence(left_km, right_km):
    x = (right_km[1] - left_km[1]) / (left_km[0] - right_km[0])
    y = right_km[0] * x + right_km[1]
    return (x, y)

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
    cropped_image, rho, theta, threshold, np.array([]), minLineLength=10, maxLineGap=30
)
line_image = display_lines(frame, lines)
cv2.imwrite("line_image.png", line_image)

combo_image = cv2.addWeighted(frame, 0.8, line_image,  1,1)
cv2.imwrite("combo_image.png", combo_image)

(averaged_points, averaged_km) = average_slope_intercept(frame, lines)
print(averaged_points[0])
print(averaged_points[1])
print(averaged_km[0], averaged_km[1])
(x, y) = convergence(averaged_km[0], averaged_km[1])
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

averaged_image = display_lines(frame, averaged_points)
cv2.imwrite("averaged_image.png", averaged_image)
averaged_combo_image = cv2.addWeighted(frame, 0.8, averaged_image, 1, 1)
cv2.imwrite("averaged_combo_image.png", averaged_combo_image)

direction_image = display_lines(frame, move_vector, color=(0, 255, 0))
cv2.imwrite("direction_image.png", direction_image)
direction_combo_image = cv2.addWeighted(averaged_combo_image, 0.8, direction_image, 1, 1)
cv2.imwrite("direction_combo_image.png", direction_combo_image)




