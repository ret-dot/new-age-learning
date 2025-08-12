import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, device='cpu', model_name='yolov8n.pt', conf_thresh=0.35):
        print('Loading YOLO model (this may be slow on CPU)...')
        self.model = YOLO(model_name)
        self.device = device
        self.model.fuse()  # attempt some small acceleration
        self.conf_thresh = conf_thresh

    def predict(self, frame):
        # frame: BGR numpy
        # return list of dict: {class, conf, bbox=[x,y,w,h]}
        results = self.model.predict(source=frame, stream=False, device=self.device, conf=self.conf_thresh, imgsz=frame.shape[:2])
        # ultralytics returns list of Results; on CPU we get a single result
        out = []
        if not results:
            return out
        res = results[0]
        boxes = res.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0].numpy()
            conf = float(box.conf[0]) if hasattr(box.conf[0], 'item') else float(box.conf[0])
            cls = int(box.cls[0]) if hasattr(box.cls[0], 'item') else int(box.cls[0])
            x1,y1,x2,y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            out.append({'class': cls, 'conf': conf, 'bbox': [float(x1), float(y1), float(w), float(h)]})
        return out