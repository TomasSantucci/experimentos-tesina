#!/usr/bin/env python3
"""
Convert timing.trajectory.csv into in.csv and out.csv
matching the format used in my_okvis2/.

in.csv:  #t_ns (col 0), in_ts  = frontend_frames_received (col 4)
out.csv: #t_ns (col 0), out_ts = consumer_state_received  (last col)
"""

import csv
import argparse
from pathlib import Path


def convert(input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    in_rows = []
    out_rows = []

    with open(input_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Identify column indices
        t_col = 0                          # #frames_original_timestamp
        in_col = header.index("frontend_frames_received")
        out_col = len(header) - 1          # last column (consumer_state_received)

        for row in reader:
            if not row:
                continue
            t_ns   = row[t_col].strip()
            in_ts  = row[in_col].strip()
            out_ts = row[out_col].strip()
            in_rows.append((t_ns, in_ts))
            out_rows.append((t_ns, out_ts))

    in_path  = output_dir / "in.csv"
    out_path = output_dir / "out.csv"

    with open(in_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#t_ns", "in_ts"])
        writer.writerows(in_rows)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#t_ns", "out_ts"])
        writer.writerows(out_rows)

    print(f"Written {len(in_rows)} rows to {in_path} and {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert timing.trajectory.csv to in/out CSVs")
    parser.add_argument("input", nargs="?",
                        default="timing.trajectory.csv",
                        help="Path to input CSV (default: timing.trajectory.csv)")
    parser.add_argument("-o", "--output-dir",
                        default=".",
                        help="Output directory for in.csv / out.csv (default: current dir)")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output_dir))
