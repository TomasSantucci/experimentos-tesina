"""
Extract the last total_time from each output_bal.log file under timing-results/
and produce a JSON file with the structure: data[run][dataset][total_time] = x

Run from the directory containing the timing-results/ folder:
    python extract_timing.py
"""

import re
import json
from pathlib import Path

BASE_DIR = Path("report/timing-results")
OUTPUT_FILE = Path("total_times_1.json")

# Matches lines like:
#   [Success] ... total_time: 4.590s, ...
# or at end: total_time: 4.590s
TOTAL_TIME_PATTERN = re.compile(r"total_time:\s*([\d.]+)s")

data = {}

for run_dir in sorted(BASE_DIR.iterdir()):
    if not run_dir.is_dir():
        continue
    run_name = run_dir.name
    data[run_name] = {}

    for dataset_dir in sorted(run_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        log_file = dataset_dir / "output_bal.log"

        if not log_file.exists():
            print(f"  [WARN] No log file in {dataset_dir}, skipping.")
            continue

        # Collect all matches; we want the last one
        matches = TOTAL_TIME_PATTERN.findall(log_file.read_text(errors="replace"))

        if not matches:
            print(f"  [WARN] No total_time found in {log_file}, skipping.")
            continue

        total_time = float(matches[-1])
        data[run_name][dataset_name] = {"total_time": total_time}
        print(f"  {run_name}/{dataset_name}: {total_time}s")

OUTPUT_FILE.write_text(json.dumps(data, indent=2))
print(f"\nSaved to {OUTPUT_FILE}")
