"""
Final 24 FPS Optimized Pipeline
--------------------------------
Implements the requested 24 fps pipeline with a smart scheduler:

- YOLO (fast) on EVERY frame (24 fps) for high recall + tracking.
- GroundingDINO + SAM on keyframes (every 24 frames → once per second) for high precision + masks.
- Places365 scene classification every 2 seconds (every 48 frames).
- SlowFast action recognition on sliding 32-frame clips sampled at 24 fps.

Outputs parquet summaries compatible with a summarize_parquet.py style workflow.

NOTE: All heavy models are optional and lazily loaded. If a dependency is missing,
      the pipeline will continue running with available components and log warnings.

Run:
    python pipeline_24fps_optimized.py \
        --video input.mp4 \
        --out_dir runs/movie1 \
        --yolo_model yolov8n.pt \
        --enable_dino --enable_sam --enable_places --enable_slowfast

Dependencies (install only what you need):
    pip install opencv-python numpy pyarrow pandas tqdm
    # For YOLO (Ultralytics):
    pip install ultralytics
    # For GroundingDINO:
    pip install groundingdino-py torch torchvision
    # For SAM:
    pip install segment-anything
    # For Places365:
    pip install torch torchvision pillow
    # For SlowFast (or use your wrapper):
    pip install torch torchvision decord

"""