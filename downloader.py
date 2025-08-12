import os
import subprocess
from pathlib import Path


def download_youtube(url: str, out_dir: str) -> str:
    """Downloads the best mp4 video using yt-dlp. Returns path to file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / '%(id)s.%(ext)s')
    cmd = [
        'yt-dlp',
        '--format', 'mp4',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '-o', out_template,
        url
    ]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)
    # find downloaded file
    # naive: return the newest file in out_dir
    files = sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        if f.suffix.lower() in ('.mp4', '.mkv', '.webm'):
            return str(f)
    raise FileNotFoundError('Download failed or no video found')