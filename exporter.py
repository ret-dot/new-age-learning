import json
from pathlib import Path
import pandas as pd

class EventExporter:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / 'events.jsonl'
        self.parquet_path = self.out_dir / 'events.parquet'

    def save_events(self, events):
        # append to jsonl
        with open(self.jsonl_path, 'a', encoding='utf8') as f:
            for e in events:
                f.write(json.dumps(e, default=str) + '\n')
        # also update parquet
        df = pd.DataFrame([{
            'track_id': e.get('track_id'),
            'class': e.get('class'),
            'start_frame': e.get('start_frame'),
            'end_frame': e.get('end_frame'),
            'ts_start': e.get('ts_start'),
            'ts_end': e.get('ts_end'),
            'motion': e.get('motion'),
            'summary': e.get('summary')
        } for e in events])
        if self.parquet_path.exists():
            df_old = pd.read_parquet(self.parquet_path)
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_parquet(self.parquet_path)
        print(f'Saved {len(events)} events')