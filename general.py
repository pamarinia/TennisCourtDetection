import torch.nn as nn
import torch
import numpy as np
import cv2
from scipy.spatial import distance

def is_point_in_image(x, y, input_width=1280, input_height=720):
    res = False
    if x and y:
        res = (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return res

def train(model, train_loader, optimizer, criterion, device, epoch, max_iters=200):
    
    model.train()

    losses = []
    max_iters = len(train_loader)

    for iter_id, batch in enumerate(train_loader):
        out = model(batch[0].float().to(device))
        ground_truth = batch[1].clone().detach().to(device).float()
        #print(ground_truth.shape)
        loss = criterion(out, ground_truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('train | epoch = {}, iter = {}/{}, loss = {}'.format(epoch, iter_id, max_iters, loss.item()))
        losses.append(loss.item())

    return np.mean(losses)

def validate(model, val_loader, criterion, device, epoch, max_dist=7):
    
    model.eval()
    losses = []

    tp, fp, tn, fn = 0, 0, 0, 0
    
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            ground_truth = torch.tensor(batch[1], dtype=torch.long, device=device)
            kps = batch[2]
            
            loss = criterion(out, ground_truth)
            losses.append(loss.item())

            output = out.argmax(dim=1).detach().cpu().numpy()
            batch_size = batch[0].shape[0]
            for i in range(batch_size):
                for kp in kps:
                    x_pred, y_pred = postprocess(output[i][kp])
                    x_gt = batch[2][kp][0]
                    y_gt = batch[2][kp][1]
                    if is_point_in_image(x_pred, y_pred) and is_point_in_image(x_gt, y_gt):
                        dist = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dist < max_dist:
                                tp +=1
                        else:
                            fp +=1
                    elif is_point_in_image(x_pred, y_pred) and not is_point_in_image(x_gt, y_gt):
                        fp +=1
                    elif not is_point_in_image(x_pred, y_pred) and is_point_in_image(x_gt, y_gt):
                        fn +=1
                    elif not is_point_in_image(x_pred, y_pred) and not is_point_in_image(x_gt, y_gt):
                        tn +=1
            
            print('val | epoch = {}, iter = {}/{}, loss = {}, tp = {}, tn = {}, fp = {}, fn = {} '.format(epoch,
                                                                                                            iter_id,
                                                                                                            len(val_loader),
                                                                                                            round(np.mean(losses), 6),
                                                                                                            tp,
                                                                                                            tn,
                                                                                                            fp,
                                                                                                            fn))
    eps = 1e-15
    precision = round(tp / (tp + fp + eps), 5)
    accuracy = round((tp + tn) / (tp + fp + tn + fn + eps), 5)
    print('precision = {}'.format(precision))
    print('accuracy = {}'.format(accuracy))

    return np.mean(losses), precision, accuracy
    

def postprocess(feature_map, scale=2):
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=2, maxRadius=30)

    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scale
            y = circles[0][0][1]*scale
    
    return x, y