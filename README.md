# Video Understanding (CPU) — Prototype

This project contains a CPU-friendly prototype pipeline that:
- Downloads a YouTube video (optional)
- Samples frames adaptively (motion-triggered)
- Runs a lightweight object detector (YOLOv8 via Ultralytics on CPU)
- Tracks objects using a lightweight SORT tracker (Kalman + IOU)
- Collapses near-duplicate frames into events
- Infers object motion direction (approach/recede/left/right/stationary)
- Exports compact event JSONL and Parquet

This is intended as a starting point; you can optimize and replace modules as needed.

## Quick start
1. Create a virtualenv and activate it.
2. Install requirements:

```
pip install -r requirements.txt
```

3. Run on a local video file:

```
python run_pipeline.py --input /path/to/video.mp4 --out_dir ./output
```

4. Or download from YouTube (yt-dlp required):

```
python run_pipeline.py --youtube https://youtu.be/xxxxx --out_dir ./output
```

## Files
- run_pipeline.py      : CLI entrypoint
- downloader.py        : YouTube download helper (yt-dlp wrapper)
- sampler.py           : Adaptive frame sampling
- detector.py          : Wrapper for YOLOv8 (CPU mode)
- tracker.py           : Lightweight SORT implementation
- dedupe.py            : Collapse near-duplicate frames / events
- motion.py            : Motion & direction inference (with camera compensation option)
- exporter.py          : Save events to jsonl / parquet
- utils.py             : shared helpers
- requirements.txt     : Python packages

"""

"""
==== requirements.txt ====
opencv-python>=4.7.0
yt-dlp>=2024.10.0
ultralytics>=8.0.0
torch>=2.0.0
numpy>=1.23
pandas>=1.5
pyarrow>=10.0
filterpy>=1.4.5
tqdm>=4.65
scikit-image>=0.20
click>=8.0
python-dateutil>=2.8
"""