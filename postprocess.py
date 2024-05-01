import sympy
from sympy import Line
import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def line_intersection(line1, line2):
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)

    if intersection:
        return intersection[0].coordinates


def refine_kps(img, x, y, crop_size=40):
    img_height, img_width = img.shape[:2]
    x_min = max(0, x - crop_size)
    x_max = min(img_width, x + crop_size)
    y_min = max(0, y - crop_size)
    y_max = min(img_height, y + crop_size)
    refined_x, refined_y = x, y

    img_crop = img[y_min:y_max, x_min:x_max]
    lines = detect_lines(img_crop)

    if len(lines) > 1:
        lines = merge_lines(lines)
        
        if len(lines) == 2:
            print(lines)
            intersection = line_intersection(lines[0], lines[1])
            if intersection:
                new_x = int(intersection[0])
                new_y = int(intersection[1])

                if new_x > 0 and new_x < img_crop.shape[0] and new_y > 0 and new_y < img_crop.shape[1]:
                    refined_x, refined_y = new_x, new_y

    return refined_x, refined_y


def detect_lines(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(img_gray, rho=1, theta=np.pi/1800, threshold = 10, minLineLength=5, maxLineGap=30)
    lines = np.squeeze(lines)
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines


def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)],
                                        dtype=np.int32)
                        mask[i + j + 1] = False
            new_lines.append(line)  
    return new_lines      

def draw_lines(image, lines, color):
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), color, 2) 
    

if __name__ == '__main__':
    lines = detect_lines(cv2.imread('input/image_crop.jpg'))
    print(lines)
    print(lines[0][0])
    lines = merge_lines(lines)
    print(lines)
    img = cv2.imread('input/image_crop.jpg')
    draw_lines(img, lines, (0, 255, 0))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
