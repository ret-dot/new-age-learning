import math


def infer_directions(events, frame_width=416, frame_height=234, radial_thresh=2.0, speed_thresh=1.0):
    """Infer direction for each event using average bbox centroids.
    Adds 'motion' key to each event: one of ['approaching','receding','left','right','stationary']
    """
    for e in events:
        bboxes = e.get('bboxes', [])
        if not bboxes:
            e['motion'] = 'unknown'
            continue
        cents = [(b[0]+b[2]/2.0, b[1]+b[3]/2.0) for b in bboxes]
        # compute simple velocity between first and last centroid
        x0,y0 = cents[0]
        x1,y1 = cents[-1]
        dx = x1 - x0
        dy = y1 - y0
        # distance to center
        cx = frame_width/2.0
        cy = frame_height/2.0
        r0 = math.hypot(x0-cx, y0-cy)
        r1 = math.hypot(x1-cx, y1-cy)
        radial = r0 - r1  # positive: moved closer to center
        # decide motion
        if abs(dx) < speed_thresh and abs(dy) < speed_thresh and abs(radial) < radial_thresh:
            motion = 'stationary'
        else:
            if abs(radial) >= radial_thresh:
                motion = 'approaching' if radial > 0 else 'receding'
            else:
                if abs(dx) > abs(dy):
                    motion = 'right' if dx > 0 else 'left'
                else:
                    motion = 'down' if dy > 0 else 'up'
        e['motion'] = motion
    return events