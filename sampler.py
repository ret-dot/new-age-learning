import cv2
import numpy as np

class AdaptiveSampler:
    """Adaptive frame sampler that increases sampling when motion is detected.
    Yields (frame_idx, timestamp_seconds, frame_resized)
    """
    def __init__(self, cap, base_fps=1.0, motion_fps=5.0, resize_width=416, motion_thresh=10.0):
        self.cap = cap
        self.base_fps = base_fps
        self.motion_fps = motion_fps
        self.resize_width = resize_width
        self.motion_thresh = motion_thresh
        self.prev_gray = None
        self.frame_idx = -1
        self.src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _resize(self, frame):
        h, w = frame.shape[:2]
        new_w = self.resize_width
        new_h = int(h * new_w / w)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def sample(self):
        step_base = max(1, int(round(self.src_fps / self.base_fps)))
        step_motion = max(1, int(round(self.src_fps / self.motion_fps)))
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_idx += 1
            ts = self.frame_idx / (self.src_fps or 30.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_mag = 0.0
            if self.prev_gray is not None:
                diff = cv2.absdiff(gray, self.prev_gray)
                motion_mag = float(diff.mean())
            self.prev_gray = gray

            # decide sampling step
            step = step_motion if motion_mag >= self.motion_thresh else step_base
            if self.frame_idx % step == 0:
                out = self._resize(frame)
                yield self.frame_idx, ts, out