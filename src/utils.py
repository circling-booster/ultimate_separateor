import os
import sys
import json
import logging
import datetime
from pathlib import Path

def setup_logger():
    logging.getLogger("audio_separator").setLevel(logging.ERROR)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_files(input_dir, extensions, target_file=None):
    if target_file and target_file.strip():
        found = list(Path(input_dir).rglob(target_file))
        return [str(f) for f in found]
    
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                audio_files.append(os.path.join(root, file))
    return audio_files

def fmt_time(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {int(s):02d}s"

def save_process_info(output_dir, filename, duration, elapsed, config):
    data = {
        "processed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": {"name": filename, "duration": round(duration, 2)},
        "stats": {"elapsed": round(elapsed, 2), "rtf": round(elapsed/duration, 3)},
        "config_snapshot": config
    }
    with open(os.path.join(output_dir, "process_info.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)