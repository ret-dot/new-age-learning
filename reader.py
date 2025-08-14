# summarize_parquet.py
import pandas as pd
from collections import defaultdict

FPS = 24  # adjust if needed
CHUNK_SECONDS = 30

df = pd.read_parquet("output/events.parquet")

# Ensure duration exists
if "duration_frames" not in df.columns:
    df["duration_frames"] = df["end_frame"] - df["start_frame"] + 1

def frames_to_time(frames):
    seconds = frames / FPS
    m, s = divmod(int(seconds), 60)
    return f"{m:02}:{s:02}"

# Group by 30s chunks
timeline = defaultdict(list)
for _, row in df.iterrows():
    start_sec = row["start_frame"] // FPS
    end_sec = row["end_frame"] // FPS
    start_chunk = start_sec // CHUNK_SECONDS
    end_chunk = end_sec // CHUNK_SECONDS

    for chunk in range(start_chunk, end_chunk + 1):
        timeline[chunk].append({
            "class_name": row.get("class_name", f"class_{row['class']}"),
            "motion": row.get("motion", "unknown")
        })

# Summarize
summaries = []
for chunk_idx in sorted(timeline.keys()):
    start_time = frames_to_time(chunk_idx * CHUNK_SECONDS * FPS)
    end_time = frames_to_time((chunk_idx + 1) * CHUNK_SECONDS * FPS)
    objects = sorted(set([item["class_name"] for item in timeline[chunk_idx]]))
    motions = sorted(set([item["motion"] for item in timeline[chunk_idx]]))

    summaries.append(
        f"From {start_time} to {end_time}, the scene contains {', '.join(objects)}. "
        f"Detected motions: {', '.join(motions)}."
    )

# Save
with open("timeline_30s.txt", "w") as f:
    f.write("\n".join(summaries))

print("Timeline saved to timeline_30s.txt")
