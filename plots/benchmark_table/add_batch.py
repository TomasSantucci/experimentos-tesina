#!/usr/bin/env python
"""
Merge runs from a batch results file (e.g. batch.json) into the data/ files
consumed by benchmark.py.

A batch file looks like:

    {
      "<metric>": {
        "<run id>": {"<sequence>": <value>, ...},
        ...
      },
      ...
    }

Each selected run is written as a new run into data/{metric}.{system}.json under
the name you choose. Metric names are mapped to the ones benchmark.py expects
(atec -> ate, rtec -> rte); everything else keeps its name.

"success" and "completion" are synthesized as 1.0 for every sequence of the run
when the batch file does not contain them (i.e. the run is assumed to have
succeeded and completed every frame).

Usage:

    # See what is inside the batch file
    ./add_batch.py batch.json --list

    # Add two runs of the basalt.full system under explicit names
    ./add_batch.py batch.json -s basalt.full -a 843c8ac3-ba-1=basalt.full.ba -a 843c8ac3-bapg-1=basalt.full.bapg

    # The system can also be given per run, as <system>:<name>
    ./add_batch.py batch.json -a 843c8ac3-1=basalt:basalt.rt -a 843c8ac3-bapg-1=basalt.full:basalt.full.bapg

    # Pick them interactively
    ./add_batch.py batch.json
"""

import argparse
import json
import sys
from pathlib import Path

# Batch metric name -> metric name used by benchmark.py / the data files
METRIC_ALIASES = {
    "atec": "ate",
    "rtec": "rte",
}

# Metrics stored as [average, standard deviation] pairs
PAIR_METRICS = {"ate", "rte", "timing"}

# Metrics benchmark.py needs but that a batch file may not provide
ASSUMED_METRICS = {"success": 1.0, "completion": 1.0}


def load_batch(path):
    with open(path, "r", encoding="utf-8") as f:
        batch = json.load(f)

    runs = {}  # run id -> {metric -> {sequence -> value}}
    for batch_metric, batch_runs in batch.items():
        metric = METRIC_ALIASES.get(batch_metric, batch_metric)
        for run_id, sequences in batch_runs.items():
            runs.setdefault(run_id, {})[metric] = sequences
    return runs


def normalize(metric, value):
    # ate/rte/timing are [avg, std] pairs in the data files, a batch file may
    # only have a single value per sequence (a single run, no deviation)
    if metric in PAIR_METRICS and not isinstance(value, list):
        return [value, 0.0]
    return value


def run_metrics(metrics, assume):
    # All metrics of a run, plus the assumed ones it does not have
    complete = {metric: dict(sequences) for metric, sequences in metrics.items()}
    if not assume:
        return complete

    sequences = sorted({ds for seqs in metrics.values() for ds in seqs})
    for metric, assumed in ASSUMED_METRICS.items():
        if metric not in complete:
            complete[metric] = {ds: assumed for ds in sequences}
    return complete


def parse_target(target, default_system):
    # "<name>" targets the default system, "<system>:<name>" overrides it
    system, sep, name = target.partition(":")
    if not sep:
        return default_system, system.strip()
    return system.strip(), name.strip()


def parse_selection(add_args, available, default_system):
    selected = []
    for arg in add_args:
        run_id, _, target = arg.partition("=")
        run_id = run_id.strip()
        system, name = parse_target(target, default_system)
        if not name or not system:
            sys.exit(f"error: --add expects <run id>=[<system>:]<name>, got {arg!r}")
        if run_id not in available:
            sys.exit(f"error: no run {run_id!r} in the batch file (see --list)")
        selected.append((run_id, system, name))
    return selected


def prompt_selection(runs, default_system):
    run_ids = list(runs)
    print("Runs in the batch file:")
    for i, run_id in enumerate(run_ids, start=1):
        metrics = ", ".join(sorted(runs[run_id]))
        n = len(next(iter(runs[run_id].values())))
        print(f"  {i}. {run_id}  ({n} sequences, metrics: {metrics})")
    print(
        f"\nFor each run, enter the name to store it under (empty to skip it)."
        f"\nThe target system is {default_system}, write <system>:<name> to use another one."
    )

    selected = []
    for run_id in run_ids:
        target = input(f"  {run_id} -> ").strip()
        if not target:
            continue
        system, name = parse_target(target, default_system)
        if not name or not system:
            sys.exit(f"error: expected [<system>:]<name>, got {target!r}")
        selected.append((run_id, system, name))
    return selected


def merge_into(data_dir, system, run_name, metrics, force, dry_run):
    # Load and update everything before writing anything, so a name collision in
    # one metric file does not leave the others half updated
    pending = []
    for metric, sequences in sorted(metrics.items()):
        path = data_dir / f"{metric}.{system}.json"

        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {metric: {}}
        data.setdefault(metric, {})

        if run_name in data[metric] and not force:
            sys.exit(f"error: {path} already has a run {run_name!r} (use --force to overwrite)")

        data[metric][run_name] = {ds: normalize(metric, v) for ds, v in sequences.items()}
        pending.append((path, data, len(sequences)))

    written = []
    for path, data, n in pending:
        if not dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                f.write("\n")
        written.append((path, n))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch_file", type=Path, help="batch results file, e.g. batch.json")
    parser.add_argument(
        "-a", "--add", action="append", default=[], metavar="RUN=[SYSTEM:]NAME",
        help="add run RUN from the batch file under the name NAME, into SYSTEM's"
             " data files if given, else --system's (repeatable)",
    )
    parser.add_argument(
        "-s", "--system", default="basalt.full",
        help="system whose data files runs are added to, i.e. data/{metric}.{system}.json"
             " (default: basalt.full)",
    )
    parser.add_argument("-d", "--data-dir", type=Path, default=None, help="data directory (default: <script dir>/data)")
    parser.add_argument("-l", "--list", action="store_true", help="list the runs in the batch file and exit")
    parser.add_argument("--no-assume", action="store_true", help="do not synthesize missing success/completion")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite runs that already exist")
    parser.add_argument("-n", "--dry-run", action="store_true", help="show what would be written, write nothing")
    args = parser.parse_args()

    data_dir = args.data_dir or Path(__file__).resolve().parent / "data"
    runs = load_batch(args.batch_file)

    if args.list:
        for run_id, metrics in runs.items():
            n = len(next(iter(metrics.values())))
            print(f"{run_id}  ({n} sequences, metrics: {', '.join(sorted(metrics))})")
        return

    if args.add:
        selected = parse_selection(args.add, runs, args.system)
    else:
        selected = prompt_selection(runs, args.system)
    if not selected:
        print("Nothing selected, nothing to do.")
        return

    targets = [(system, name) for _, system, name in selected]
    if len(set(targets)) != len(targets):
        sys.exit(f"error: duplicate targets in the selection: {targets}")

    for run_id, system, name in selected:
        metrics = run_metrics(runs[run_id], assume=not args.no_assume)
        written = merge_into(data_dir, system, name, metrics, args.force, args.dry_run)
        print(f"{run_id} -> {name} ({system})")
        for path, n in written:
            print(f"  {'would write' if args.dry_run else 'wrote'} {path} ({n} sequences)")


if __name__ == "__main__":
    main()
