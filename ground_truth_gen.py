import numpy as np
import cv2

def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = 255 * np.exp(-(x**2 + y**2)/float(2*variance))
    return g.astype(int)

def create_gt_images(kps, size, variance, input_height, input_width, output_height, output_width):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    heatmaps = np.zeros((len(kps), input_height, input_width), dtype=np.uint8)
    output = []
    for num in range(len(kps)):  
        x, y = kps[num][0], kps[num][1]
        for i in range(-size, size+1):
            for j in range(-size, size+1):
                if x + i >= 0 and x + i < input_width and y + j >= 0 and y + j < input_height:
                    heatmaps[num][y + j, x + i] = gaussian_kernel_array[j + size, i + size]
        output.append(cv2.resize(heatmaps[num], (output_width, output_height)))

    return output
