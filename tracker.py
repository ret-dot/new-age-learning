import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes in [x1,y1,x2,y2] format.
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, conf, class_id, frame_idx):
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        for i in range(4):
            self.kf.F[i, i+3] = 1.0

        self.kf.H = np.zeros((4, 7))
        self.kf.H[0, 0] = 1.0
        self.kf.H[1, 1] = 1.0
        self.kf.H[2, 2] = 1.0
        self.kf.H[3, 3] = 1.0

        self.kf.P[4:, 4:] *= 1000.  # high uncertainty for velocities
        self.kf.P *= 10.
        self.kf.R[2:, 2:] *= 10.

        self.kf.x[:4] = np.array(bbox).reshape((4, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.conf = conf
        self.class_id = class_id
        self.first_frame = frame_idx

    def update(self, bbox):
        """Updates the state vector with observed bbox."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array(bbox))
    
    def predict(self):
        """Advances the state vector and returns the predicted bounding box estimate."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:4].reshape(-1))
        return self.history[-1]
    
    def get_state(self):
        """Returns the current bounding box estimate."""
        return self.kf.x[:4].reshape(-1)

class SortTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Params:
          detections - list of dicts with keys ['bbox', 'conf', 'class', 'frame_idx', 'ts']
        Returns:
          List of dicts: [{'track_id', 'bbox', 'class', 'conf', 'frame_idx', 'ts'}]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        # Get predicted locations from existing trackers
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :4] = pos
            trks[t, 4] = trk.id
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections, trks)

        # Update matched trackers
        for det_idx, trk_idx in matched:
            det = detections[det_idx]
            self.trackers[trk_idx].update(det['bbox'])
            self.trackers[trk_idx].conf = det['conf']
            self.trackers[trk_idx].class_id = det['class']

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            trk = KalmanBoxTracker(det['bbox'], det['conf'], det['class'], det['frame_idx'])
            self.trackers.append(trk)

        # Prepare return list
        for trk in self.trackers:
            if (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                ret.append({
                    'track_id': trk.id,
                    'bbox': trk.get_state().tolist(),
                    'class': trk.class_id,
                    'conf': trk.conf,
                    'frame_idx': trk.first_frame,
                    'ts': detections[0]['ts'] if detections else None
                })

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return ret

    def _associate_detections_to_trackers(self, detections, trks):
        """
        Assigns detections to tracked objects based on IoU.
        Returns 3 lists of matches, unmatched_detections, unmatched_trackers.
        """
        if len(trks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(trks)))

        iou_matrix = np.zeros((len(detections), len(trks)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det['bbox'], trk[:4])

        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trks)))

        while True:
            if iou_matrix.size == 0:
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[i, j] < self.iou_threshold:
                break
            matched_indices.append((i, j))
            unmatched_detections.remove(i)
            unmatched_trackers.remove(j)
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        return matched_indices, unmatched_detections, unmatched_trackers
