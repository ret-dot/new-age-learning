import cv2
import math

def open_video_capture(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video: ' + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, frame_count