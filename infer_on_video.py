import cv2
import numpy as np
import torch
from postprocess import refine_kps

from general import postprocess
from scipy.spatial import distance

from tracknet import CourtTrackNet

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def infer_model(frames, model):
    keypoints_track = []
    for frame in frames:
        img = cv2.resize(frame, (640, 360))
        img = img.astype(np.float32) / 255.0
        img = np.rollaxis(img, 2, 0)
        img = np.expand_dims(img, axis=0)
        input = torch.from_numpy(img).float().to(device)

        out = model(input)
        output = out.detach().cpu().numpy()
        keypoints = []
        for hm in output[0]:
            x_pred, y_pred = postprocess(hm)
            keypoints.append((x_pred, y_pred))
        #print(keypoints)
        keypoints_track.append(keypoints)

    return keypoints_track


def remove_outliers(ball_track, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            
            ball_track[i-1] = (None, None)
    return ball_track


def write_video(frames, keypoints_track, path_output_video, fps):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num]
        for kp in keypoints_track[num]:
            x = int(kp[0])
            y = int(kp[1])
            frame = cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        out.write(frame)
    out.release()


if __name__ == '__main__':

    model = CourtTrackNet(out_channels=15)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load('models/model_tennis_court_det.pt'))
    model = model.to(device)
    model.eval()

    frames, fps = read_video('input/Med_Djo_cut.mp4')
    keypoints_track = infer_model(frames, model)
    
    for i, frame in enumerate(frames):
        #print(keypoints_track[i])
        keypoints = keypoints_track[i]
        for j, kp in enumerate(keypoints):
            x, y = kp
            x, y = refine_kps(frame, int(x), int(y))
            keypoints_track[i][j] = x, y
        #print(keypoints_track[i])
        



    write_video(frames, keypoints_track, 'outputs/Med_Djo_cut_tracked.avi', fps)




