# improved_detector.py
import torch
from ultralytics import YOLO

class ImprovedDetector:
    def __init__(self, model_path="yolov8x.pt", device="cpu", conf_thresh=0.4):
        """
        Use a heavy YOLO model for higher accuracy.
        You can swap 'yolov8x.pt' with 'yolov9c.pt' or any trained model.
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.conf_thresh = conf_thresh

    def scene_classify(self, frame):
        """
        Placeholder scene classifier.
        Replace with a real classifier like Places365 if needed.
        """
        return "unknown"

    def label_cleanup(self, detections):
        """
        Remove very low confidence or duplicate labels.
        """
        seen_labels = set()
        cleaned = []
        for det in detections:
            label = det["class_name"]
            if det["conf"] >= self.conf_thresh and label not in seen_labels:
                seen_labels.add(label)
                cleaned.append(det)
        return cleaned

    def predict(self, frame):
        """
        Run YOLO, clean labels, return final detections.
        """
        results = self.model.predict(
            frame, 
            conf=self.conf_thresh, 
            device=self.device, 
            verbose=False
        )

        final_dets = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = self.model.names[cls_id]
                det = {
                    "class": cls_id,
                    "class_name": class_name,
                    "conf": float(box.conf),
                    "bbox": box.xywh[0].tolist()
                }
                final_dets.append(det)

        # Remove duplicates / junk
        final_dets = self.label_cleanup(final_dets)
        return final_dets


if __name__ == "__main__":
    import cv2
    det = ImprovedDetector(model_path="yolov8x.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture("sample.mp4")
    ret, frame = cap.read()
    if ret:
        detections = det.predict(frame)
        print(detections)
    cap.release()
