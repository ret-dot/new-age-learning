"""Entry point for the CPU-based video understanding pipeline."""
import os
import sys
import argparse
from pathlib import Path
from downloader import download_youtube
from sampler import AdaptiveSampler
from detector import ImprovedDetector
from tracker import SortTracker
from dedupe import collapse_events
from motion import infer_directions
from exporter import EventExporter
from utils import open_video_capture


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, help='Path to local video file')
    p.add_argument('--youtube', type=str, help='YouTube URL to download')
    p.add_argument('--out_dir', type=str, default='./output')
    p.add_argument('--sample_fps', type=float, default=2.0, help='Base sampling fps')
    p.add_argument('--motion_fps', type=float, default=6.0, help='Sampling fps during motion')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--resize_width', type=int, default=416)
    p.add_argument('--min_conf', type=float, default=0.35)
    p.add_argument('--iou_thresh', type=float, default=0.85)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    video_path = None
    if args.youtube:
        print('Downloading youtube video...')
        video_path = download_youtube(args.youtube, args.out_dir)
    elif args.input:
        video_path = args.input
    else:
        print('Provide --input or --youtube')
        sys.exit(1)

    cap, fps, frame_count = open_video_capture(video_path)
    print(f'Opened video {video_path} fps={fps} frames={frame_count}')

    sampler = AdaptiveSampler(cap, base_fps=args.sample_fps, motion_fps=args.motion_fps)
    detector = Detector(device=args.device, model_name='yolov8n.pt', conf_thresh=args.min_conf)
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    exporter = EventExporter(args.out_dir)

    frames_buffer = []
    detections_buffer = []
    tracks_buffer = []

    for frame_idx, ts, frame in sampler.sample():
        # frame is resized to width provided by sampler
        dets = detector.predict(frame)  # returns list of {class,conf,bbox}
        for d in dets:
            d.update({'frame_idx': frame_idx, 'ts': ts})
        tracks = tracker.update(dets)
        # tracks => list of dict with track_id, bbox, class, conf, frame_idx, ts
        frames_buffer.append((frame_idx, ts))
        detections_buffer.extend(dets)
        tracks_buffer.extend(tracks)

        # periodically flush to exporter
        if len(tracks_buffer) > 1000 or frame_idx == frame_count - 1:
            print(f'Processing {len(tracks_buffer)} track items...')
            events = collapse_events(tracks_buffer, iou_thresh=args.iou_thresh)
            events = infer_directions(events)
            exporter.save_events(events)
            frames_buffer.clear()
            detections_buffer.clear()
            tracks_buffer.clear()

    cap.release()
    print('Done. Outputs in', args.out_dir)

if __name__ == '__main__':
    main()