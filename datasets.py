from torch.utils.data import Dataset
from ground_truth_gen import create_gt_images
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import torch


class CourtDataset(Dataset):

    def __init__(self, mode):
        
        self.path_dataset = 'datasets/court_dataset'
        
        assert mode in ['train', 'val'], 'incorrect mode'
        
        with open(os.path.join(self.path_dataset, 'data_{}.json'.format(mode)), 'r') as f:
            self.data = json.load(f)
        print('mode = {}, samples = {}'.format(mode, len(self.data)))
        
        self.input_height = 720
        self.input_width = 1280
        self.output_height = 360
        self.output_width = 640
        self.size = 20
        self.variance = 10
        
        self.gaussian_kernel_array = self.gaussian_kernel(self.size, self.variance)


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        img_name = self.data[index]['id'] + '.png'
        kps = self.data[index]['kps']

        img_path = os.path.join(self.path_dataset, 'images', img_name)
        input = self.get_input(img_path)
        
        output = self.create_heatmaps(kps)

        return input, output, kps
    
    
    def get_input(self, img_path):
        # Resize the images to (360, 640) to speed up the training
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.output_width, self.output_height))
        
        # Normalize the images
        img = img.astype(np.float32) / 255.0
        # Change the shape to (C, H, W)
        img = np.rollaxis(img, 2, 0)

        return img
    

    def gaussian_kernel(self, size, variance):
        x, y = np.mgrid[-size:size+1, -size:size+1]
        g = 255 * np.exp(-(x**2 + y**2)/float(2*variance))
        return g.astype(int)

    def create_heatmaps(self, kps):
        heatmaps = np.zeros((len(kps), self.input_height, self.input_width), dtype=np.uint8)
        for num in range(len(kps)):  
            x, y = kps[num][0], kps[num][1]
            for i in range(-self.size, self.size+1):
                for j in range(-self.size, self.size+1):
                    if x + i >= 0 and x + i < self.input_width and y + j >= 0 and y + j < self.input_height:
                        heatmaps[num][y + j, x + i] = self.gaussian_kernel_array[j + self.size, i + self.size]
        resized_heatmaps = []
        for hm in heatmaps:
            resized_hm = cv2.resize(hm, (self.output_width, self.output_height))
            resized_hm = np.reshape(resized_hm, (self.output_height * self.output_width))
            resized_heatmaps.append(resized_hm)
        
        return np.array(resized_heatmaps)

    

if __name__ == '__main__':
    dataset = CourtDataset('train')
    input, output, kps = dataset[0]
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(input.transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(output[0], cmap='gray')
    plt.show()
