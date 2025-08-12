import math
from collections import defaultdict


def bbox_centroid(bbox):
    x,y,w,h = bbox
    return (x + w/2.0, y + h/2.0)


def iou_bb(bb1, bb2):
    x1,y1,w1,h1 = bb1
    x2,y2,w2,h2 = bb2
    xx1 = max(x1,x2)
    yy1 = max(y1,y2)
    xx2 = min(x1+w1, x2+w2)
    yy2 = min(y1+h1, y2+h2)
    w = max(0, xx2-xx1)
    h = max(0, yy2-yy1)
    inter = w*h
    a1 = w1*h1
    a2 = w2*h2
    return inter / (a1 + a2 - inter + 1e-6)


def collapse_events(tracks_list, iou_thresh=0.85, centroid_thresh_frac=0.02):
    """Collapse near-duplicate frames into events.
    tracks_list: list of track dicts with track_id, bbox, class, conf, frame_idx, ts
    Returns list of events: each event aggregates repeated frames for same track.
    """
    # group by track_id
    by_track = defaultdict(list)
    for t in tracks_list:
        by_track[t['track_id']].append(t)
    events = []
    for track_id, seq in by_track.items():
        seq = sorted(seq, key=lambda x: x['frame_idx'])
        cur_event = None
        for item in seq:
            if cur_event is None:
                cur_event = {'track_id': track_id, 'class': item['class'], 'conf': item['conf'], 'start_frame': item['frame_idx'], 'end_frame': item['frame_idx'], 'bboxes':[item['bbox']], 'ts_start': item.get('ts', None), 'ts_end': item.get('ts', None)}
            else:
                prev_bbox = cur_event['bboxes'][-1]
                i = iou_bb(prev_bbox, item['bbox'])
                # centroid distance relative to frame width/height (we'll use bbox width)
                cx1,cy1 = bbox_centroid(prev_bbox)
                cx2,cy2 = bbox_centroid(item['bbox'])
                dx = abs(cx2-cx1)
                # threshold: small fraction of bbox width
                w = (prev_bbox[2] + item['bbox'][2]) / 2.0
                if i >= iou_thresh or dx <= centroid_thresh_frac * w:
                    # treat as same event; extend
                    cur_event['end_frame'] = item['frame_idx']
                    cur_event['bboxes'].append(item['bbox'])
                    cur_event['conf'] = max(cur_event['conf'], item['conf'])
                    cur_event['ts_end'] = item.get('ts', cur_event['ts_end'])
                else:
                    # finalize previous
                    events.append(cur_event)
                    cur_event = {'track_id': track_id, 'class': item['class'], 'conf': item['conf'], 'start_frame': item['frame_idx'], 'end_frame': item['frame_idx'], 'bboxes':[item['bbox']], 'ts_start': item.get('ts', None), 'ts_end': item.get('ts', None)}
        if cur_event is not None:
            events.append(cur_event)
    # simple sort
    events = sorted(events, key=lambda e: e['ts_start'] if e['ts_start'] is not None else e['start_frame'])
    # add simple summary tokens
    for e in events:
        e['duration_frames'] = e['end_frame'] - e['start_frame'] + 1
        e['avg_bbox'] = [sum(v)/len(v) for v in zip(*e['bboxes'])]
        e['summary'] = f"{e['duration_frames']} frames of class_{e['class']}"
    return events
