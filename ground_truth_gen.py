import numpy as np
import os
import json
import cv2

def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = 255 * np.exp(-(x**2 + y**2)/float(2*variance))
    return g.astype(int)

def create_gt_images(output_path, size, variance, height, width):
    path_train_dataset = 'datasets/court_dataset/data_train.json'
    with open(path_train_dataset, 'r') as f:
        data_train = json.load(f)
    path_train_dataset = 'datasets/court_dataset/data_val.json'
    with open(path_train_dataset, 'r') as f:
        data_val = json.load(f)
    print(data_train[0]['kps'])
    data_train = [[x['id'], x['kps']] for x in data_train]
    data_val = [[x['id'], x['kps']] for x in data_val]

    datas = data_train + data_val

    print(data_train[0])
    # Create the directory structure for game
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gaussian_kernel_array = gaussian_kernel(size, variance)
    for data in datas:  
        img_name = data[0] + '.png'
        kps = data[1]
        heatmap = np.zeros((height, width), dtype=np.uint8)
        for kp in kps:
            x, y = kp
            
            for i in range(-size, size+1):
                for j in range(-size, size+1):
                    if x + i >= 0 and x + i < width and y + j >= 0 and y + j < height:
                        heatmap[y + j, x + i] = gaussian_kernel_array[j + size, i + size]
        cv2.imwrite(os.path.join(output_path, img_name), heatmap)

if __name__ == '__main__':
    create_gt_images('datasets/court_dataset/ground_truth', 5, 1, 720, 1280)