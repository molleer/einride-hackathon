import cv2
import numpy as np


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

def average_slope_intercept(lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return [[-1000, 0], [1000, 160000]]
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
    if (left_fit_average is None) or np.any(np.isnan(left_fit_average)):
        left_fit_average = [-1000, 0]
    if (right_fit_average is None) or np.any(np.isnan(right_fit_average)):
        right_fit_average = [1000, 160000]
    return [left_fit_average, right_fit_average]

def convergence(left_km, right_km):
    x = (right_km[1] - left_km[1]) / (left_km[0] - right_km[0])
    y = right_km[0] * x + right_km[1]
    return (x, y)
