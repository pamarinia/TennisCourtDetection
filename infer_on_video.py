import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps

from general import postprocess
from scipy.spatial import distance

from tracknet import CourtTrackNet

def read_video(video_path):
    """
    This function reads a video from video_path and returns the frames and the frames per second.

    Args:
        video_path: A string representing the path to the video.
    
    Returns:
        frames: A list of numpy arrays representing the frames of the video.
        fps: An integer representing the frames per second of the video.
    """
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


def write_video(imgs_new, fps, path_output_video):
    """
    This function writes the frames to a video.

    Args:
        imgs_new: A list of numpy arrays representing the frames.
        fps: An integer representing the frames per second of the video.
        path_output_video: A string representing the path to the output video.
    """
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for img in imgs_new:
        out.write(img)
    out.release()


def add_circle_frame(frame, keypoints_track):
    """
    This function adds circles to the keypoints of the frames.

    Args:
        keypoints_track: A list of lists. Each list contains the keypoints of a frame.
    
    Returns:
        frames: A list of numpy arrays representing the frames with the keypoints.
    """
    for kp in keypoints_track:
        x = int(kp[0])
        y = int(kp[1])
        frame = cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return frame

def infer_model(frames, model):
    """
    This function takes a list of frames and a model and returns the keypoints of the frames.
    
    Args:
        frames: A list of numpy arrays representing the frames.
        model: A PyTorch model.
    
    Returns:
        keypoints_track: A list of lists. Each list contains the keypoints of a frame.
    """
    keypoints_track = []
    for frame in tqdm(frames):
        img = cv2.resize(frame, (640, 360))
        img = img.astype(np.float32) / 255.0
        img = np.rollaxis(img, 2, 0)
        img = torch.tensor(img).unsqueeze(0)
        input = img.float().to(device)

        out = model(input)[0]
        preds = F.sigmoid(out).detach().cpu().numpy()

        keypoints = []
        for kps_num in range(14):
            heatmap = (preds[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap)
            if kps_num not in [8, 9, 12]:
                x_pred, y_pred = refine_kps(frame, int(x_pred), int(y_pred), kps_num)
            keypoints.append((x_pred, y_pred))
        #print(keypoints)

        matrix_trans = get_trans_matrix(keypoints)
        if matrix_trans is not None:
            keypoints = cv2.perspectiveTransform(refer_kps, matrix_trans)
            keypoints = [np.squeeze(kp) for kp in keypoints]
        
        keypoints_track.append(keypoints)

    return keypoints_track


def remove_outliers(ball_track, dists, max_dist=100):
    """
    This function removes the outliers from the ball_track.

    Args:
        ball_track: A list of tuples. Each tuple contains the (x, y) coordinates of the ball.
        dists: A list of distances between the keypoints.
        max_dist: An integer representing the maximum distance between the keypoints.   
    
    Returns:
        ball_track: A list of tuples. Each tuple contains the (x, y) coordinates of the ball.
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            
            ball_track[i-1] = (None, None)
    return ball_track


if __name__ == '__main__':

    model = CourtTrackNet(out_channels=15)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.load_state_dict(torch.load('models/model_tennis_court_det.pt'))
    model.eval()

    frames, fps = read_video('input/Med_Djo_cut.mp4')
    keypoints_track = infer_model(frames, model)
    
    for frame, keypoints in zip(frames, keypoints_track):
        frame = add_circle_frame(frame, keypoints)
    write_video(frames, fps, 'outputs/Med_Djo_cut_tracked.avi')




