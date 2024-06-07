import sympy
from sympy import Line
import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def line_intersection(line1, line2):
    """
    This function takes two lines and returns the intersection point of the lines.
    
    Args:
        line1: A list of four integers representing the coordinates of the first line.
        line2: A list of four integers representing the coordinates of the second line.
    
    Returns:
        intersection: A list of two integers representing the intersection point of the lines.
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)

    if intersection:
        #print('Intersection found')
        return intersection[0].coordinates
    else:
        pass
        #print('No intersection found')


def refine_kps(img, x, y, number, crop_size=25):
    """
    This function refines the keypoints by detecting the lines around the keypoints and finding the intersection point of the lines.

    Args:
        img: A numpy array representing the image.
        x: An integer representing the x-coordinate of the keypoint.
        y: An integer representing the y-coordinate of the keypoint.
        crop_size: An integer representing the size of the crop around the keypoint.
    
    Returns:   
        refined_x: An integer representing the refined x-coordinate of the keypoint.
        refined_y: An integer representing the refined y-coordinate of the keypoint.
    """
    img_height, img_width = img.shape[:2]
    x_min = max(0, x - crop_size)
    x_max = min(img_width, x + crop_size)
    y_min = max(0, y - crop_size)
    y_max = min(img_height, y + crop_size)    
    
    img_crop = img[y_min:y_max, x_min:x_max]
    
    center_x = img_crop.shape[1] // 2
    center_y = img_crop.shape[0] // 2

    refined_x, refined_y = x, y

    lines = detect_lines(img_crop)

    
    if len(lines) > 1:

        lines = merge_lines(lines)
        #print(len(lines))
        if len(lines) == 2:
            #img_crop = cv2.line(img_crop, (lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (255, 255, 0), 2)
            #img_crop = cv2.line(img_crop, (lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]), (0, 255, 255), 2)
            intersection = line_intersection(lines[0], lines[1])
            if intersection:
                new_x = int(intersection[0])
                new_y = int(intersection[1])

                if new_x > 0 and new_x < img_crop.shape[0] and new_y > 0 and new_y < img_crop.shape[1]:
                    refined_x, refined_y = new_x, new_y
                    #img_crop = cv2.circle(img_crop, (refined_x, refined_y), radius=4, color=(255, 0, 0), thickness=-1)
                    refined_x = x - center_x + refined_x
                    refined_y = y - center_y + refined_y
        elif len(lines) < 2:
            pass
            #print('Not enough lines found')
        else:
            pass
            #print('More than two lines found')
    else:
        pass
        #print('No lines found')


    #img_crop = cv2.circle(img_crop, (center_x, center_y), radius=6, color=(0, 0, 0), thickness=-1)
    #img_crop = cv2.circle(img_crop, (refined_x, refined_y), radius=4, color=(255, 0, 0), thickness=-1)
    #cv2.imwrite(f'input/image_crop_{number}.jpg', img_crop)

    return refined_x, refined_y


def detect_lines(img):
    """
    This function detects lines in the image using HoughLinesP.
    
    Args:
        img: A numpy array representing the image.
    
    Returns:
        lines: A list of lines. Each line is a list of four integers representing the coordinates of the line.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(img_gray, rho=1, theta=np.pi/1800, threshold = 10, minLineLength=5, maxLineGap=20)
    lines = np.squeeze(lines)
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines

def angle_between_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    return np.abs(np.degrees((angle1 - angle2) % (2 * np.pi)))

def merge_lines(lines):
    """
    This function merges the lines that are close to each other.
    
    Args:
        lines: A list of lines. Each line is a list of four integers representing the coordinates of the line.
    
    Returns:
        new_lines: A list of lines. Each line is a list of four integers representing the coordinates of the line.
    """
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    """for i, line in enumerate(lines):
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
            new_lines.append(line)  """

    # Find two lines that are approximately perpendicular
    min_diff = float('inf')
    for i in range(0,len(lines)):
        line_i = lines[i]
        for j in range(i,len(lines)):
            diff = angle_between_lines(line_i, lines[j])
            if diff > 30 and diff < 140:
                #print(diff)
                return [lines[i], lines[j]]

    return new_lines      


def draw_lines(image, lines, color):
    """
    This function draws the lines on the image.
    
    Args:
        image: A numpy array representing the image.
        lines: A list of lines. Each line is a list of four integers representing the coordinates of the line.
        color: A tuple of three integers representing the color of the line.
    
    Returns:
        image: A numpy array representing the image with the lines drawn on it.
    """
    i = 0
    for line in lines:
        colors = (10*i, 20*i, 30*i)
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), colors, 2) 
        i += 10

if __name__ == '__main__':
    lines = detect_lines(cv2.imread('input/image_crop.jpg'))
    #print(lines)
    #print(lines[0][0])
    #lines = merge_lines(lines)
    #print(lines)
    img = cv2.imread('input/image_crop.jpg')
    #draw_lines(img, lines, (0, 255, 0))
    
    x, y = refine_kps(img, 20, 30, crop_size=40)
    cv2.circle(img, (20, 30), radius=3, color=(0, 0, 0), thickness=-1)
    cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
