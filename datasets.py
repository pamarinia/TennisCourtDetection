from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json


class CourtDataset(Dataset):

    def __init__(self, mode, input_height=360, input_width=640):
        
        self.path_dataset = 'datasets/court_dataset'
        
        assert mode in ['train', 'val'], 'incorrect mode'
        
        with open(os.path.join(self.path_dataset, 'data_{}.json'.format(mode)), 'r') as f:
            self.data = json.load(f)
        
        print('mode = {}, samples = {}'.format(mode, len(self.data)))
        
        self.height = input_height
        self.width = input_width


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        img_name = self.data[index]['id'] + '.png'
        kps = self.data[index]['kps']

        img_path = os.path.join(self.path_dataset, 'images', img_name)
        ground_truth_path = os.path.join(self.path_dataset, 'ground_truth', img_name)
        input = self.get_input(img_path)
        output = self.get_output(ground_truth_path)

        return input, output, kps
    
    
    def get_input(self, img_path):
        # Resize the images to (360, 640) to speed up the training
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.width, self.height))
        
        # Normalize the images
        img = img.astype(np.float32) / 255.0
        # Change the shape to (C, H, W)
        img = np.rollaxis(img, 2, 0)

        return img
    
    
    def get_output(self, ground_truth_path):
        gt = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (self.width, self.height))
        gt = np.reshape(gt, (self.height * self.width))

        return gt
    

