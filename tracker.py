# Lightweight SORT tracker implementation using a Kalman filter from filterpy
import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[0]+bb_test[2], bb_gt[0]+bb_gt[2])
    yy2 = np.minimum(bb_test[1]+bb_test[3], bb_gt[1]+bb_gt[3])
    w = np.maximum(0., xx2-xx1)
    h = np.maximum(0., yy2-yy1)
    inter = w*h
    area1 = bb_test[2]*bb_test[3]
    area2 = bb_gt[2]*bb_gt[3]
    o = inter / (area1 + area2 - inter + 1e-6)
    return o

class Track:
    def __init__(self, bbox, track_id, frame_idx, conf, cls):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.
        self.kf.F = np.array([[1,0,0,0,dt,0,0],
                              [0,1,0,0,0,dt,0],
                              [0,0,1,0,0,0,dt],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.x[:4] = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0,0,0])
        self.kf.P *= 10.
        self.time_since_update = 0
        self.id = track_id
        self.hits = 1
        self.age = 0
        self.history = []
        self.class_id = cls
        self.conf = conf
        self.last_frame = frame_idx

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x

    def update(self, bbox, conf, frame_idx):
        z = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf
        self.last_frame = frame_idx
        self.history.append((frame_idx, bbox))

    def get_state(self):
        x = self.kf.x
        return [float(x[0]), float(x[1]), float(x[2]), float(x[3])]

class SortTracker:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        # detections: list of {'bbox':[x,y,w,h], 'class':cls, 'conf':conf, 'frame_idx':..}
        # predict existing
        for t in self.tracks:
            t.predict()
        N = len(self.tracks)
        M = len(detections)
        matches, unmatched_dets, unmatched_trks = [], list(range(M)), list(range(N))
        if N>0 and M>0:
            iou_mat = np.zeros((N,M), dtype=np.float32)
            for i,t in enumerate(self.tracks):
                trk_state = t.get_state()
                for j,d in enumerate(detections):
                    iou_mat[i,j] = iou(trk_state, d['bbox'])
            # greedy matching
            for _ in range(min(N,M)):
                i,j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                if iou_mat[i,j] < self.iou_threshold:
                    break
                matches.append((i,j))
                iou_mat[i,:] = -1
                iou_mat[:,j] = -1
                unmatched_trks.remove(i)
                unmatched_dets.remove(j)
        # update matched
        outputs = []
        for (i,j) in matches:
            trk = self.tracks[i]
            d = detections[j]
            trk.update(d['bbox'], d['conf'], d['frame_idx'])
            outputs.append({'track_id': trk.id, 'bbox': trk.get_state(), 'class': trk.class_id, 'conf': trk.conf, 'frame_idx': d['frame_idx'], 'ts': d.get('ts', None)})
        # create new tracks for unmatched detections
        for j in unmatched_dets:
            d = detections[j]
            trk = Track(d['bbox'], self.next_id, d['frame_idx'], d['conf'], d['class'])
            self.next_id += 1
            self.tracks.append(trk)
            outputs.append({'track_id': trk.id, 'bbox': trk.get_state(), 'class': trk.class_id, 'conf': trk.conf, 'frame_idx': d['frame_idx'], 'ts': d.get('ts', None)})
        # remove dead tracks
        removed = []
        for trk in list(self.tracks):
            if trk.time_since_update > self.max_age:
                if trk.hits >= self.min_hits:
                    # finalize
                    removed.append(trk)
                self.tracks.remove(trk)
        # return outputs (active and newly created), simple format
        return output