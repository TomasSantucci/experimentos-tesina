#!/usr/bin/env python
# pylint: disable-all
"""
Generate the benchmark table that benchmark.py displays, once per dataset
group, and write each one as its own PDF inside a single output folder.

benchmark.py plots a single group at a time (whatever DS_PREFIX is set to) and
shows it on screen. This script loops over all the groups and writes them to
all_benchmarks/<PREFIX>.pdf instead, one file per group.

    ./all_benchmarks.py                          # -> all_benchmarks/{MI,MG,...}.pdf
    ./all_benchmarks.py -p MI MG MO              # only those groups
    ./all_benchmarks.py -o tables                # -> tables/{MI,MG,...}.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")  # Nothing is shown on screen, everything goes to the PDF

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BATCH_DIR = Path(__file__).resolve().parent / "data"
PREFIXES = ["MI", "MG", "MO", "EMH", "EV1", "EV2", "TC", "TM", "TO", "TR", "TS"]

# Where each column comes from. A column is either a single deterministic run
# (data/{metric}.{system}.json, run <run>) or the best of several runs, picked
# per sequence by select_run() the same way benchmark.py does it.
#
# These are the fully postprocessed (non causal) variants, i.e. CAUSAL = False
# in benchmark.py. Note that okvis2.vio only covers the MSD sequences, so the
# EuRoC and TUM-VI groups need one of the okvis2.slam* variants here.
COLUMNS = [
    {"name": "Basalt", "system": "basalt", "run": "basalt.det"},
    {"name": "BasaltLCR", "system": "basalt.full", "run": "basalt.full.pg.det"},
    {"name": "OKVIS2", "system": "okvis2.slamfull", "runs": [1, 2, 3]},
    {"name": "ORB-SLAM3", "system": "orbslam3.ba", "runs": [1, 2, 3]},
]

UNK = -1234567890  # (?) Represents an unknown value, latest data doesnt have it
NA = -1234567891  # (—)

DATASETS_MSD = ["MIO01","MIO02","MIO03","MIO04","MIO05","MIO06","MIO07","MIO08","MIO09","MIO10","MIO11","MIO12","MIO13","MIO14","MIO15","MIO16","MIPB01","MIPB02","MIPB03","MIPB04","MIPB05","MIPB06","MIPB07","MIPB08","MIPP01","MIPP02","MIPP03","MIPP04","MIPP05","MIPP06","MIPT01","MIPT02","MIPT03","MGO01","MGO02","MGO03","MGO04","MGO05","MGO06","MGO07","MGO08","MGO09","MGO10","MGO11","MGO12","MGO13","MGO14","MGO15","MOO01","MOO02","MOO03","MOO04","MOO05","MOO06","MOO07","MOO08","MOO09","MOO10","MOO11","MOO12","MOO13","MOO14","MOO15","MOO16"]  # fmt: skip
DATASETS_EUROC = ["EMH01","EMH02","EMH03","EMH04","EMH05","EV101","EV102","EV103","EV201","EV202","EV203"]  # fmt: skip
DATASETS_TUMVI = ["TC1","TC2","TC3","TC4","TC5","TM1","TM2","TM3","TM4","TM5","TM6","TO1","TO2","TO3","TO4","TO5","TO6","TO7","TO8","TR1","TR2","TR3","TR4","TR5","TR6","TS1","TS2","TS3"]  # fmt: skip
DATASETS = DATASETS_MSD + DATASETS_EUROC + DATASETS_TUMVI

SYSTEMS = [column["name"] for column in COLUMNS]

M_SCALER = 100  # 100 for cm, 1 for m
ATE_DIVERGE_FROM = 10 * M_SCALER  # 10m
RTE_DIVERGE_FROM = 0.1 * M_SCALER  # 10cm
SUCCESS_FROM = 98  # From 98% we just print the checkmark

U = abs(UNK)
N = abs(NA)


# ---------------------------------------------------------------------------
# Loading and run selection (same rules as benchmark.py, per dataset group)
# ---------------------------------------------------------------------------


def load_storage(systems, datasets):
    # Load all system numbers into a dict storage
    storage = {}
    metrics = ["ate", "rte", "success", "completion"]
    load_files = [BATCH_DIR / f"{metric}.{system}.json" for metric in metrics for system in systems]
    for load_file in load_files:
        with open(load_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for metric, runs in loaded.items():  # Merge loaded files
                if metric not in storage:
                    storage[metric] = {}
                for run_name, run in runs.items():
                    for ds in datasets:
                        if ds in run and run[ds] is None:  # Likely crash
                            run[ds] = [NA, NA] if metric in ["ate", "rte"] else NA
                        if ds not in run:  # Unknown
                            run[ds] = [UNK, UNK] if metric in ["ate", "rte"] else UNK
                storage[metric] |= runs

    return storage


def select_runs(storage, selected):
    return {
        "completion": {ds: storage["completion"][r][ds] for ds, r in selected.items()},
        "success": {ds: storage["success"][r][ds] for ds, r in selected.items()},
        "ates_avg": {ds: storage["ate"][r][ds][0] for ds, r in selected.items()},
        "ates_std": {ds: storage["ate"][r][ds][1] for ds, r in selected.items()},
        "rtes_avg": {ds: storage["rte"][r][ds][0] for ds, r in selected.items()},
        "rtes_std": {ds: storage["rte"][r][ds][1] for ds, r in selected.items()},
        "selected": selected,
    }


def select_run(storage, ds, runs):
    # Map each run to its success for the given dataset
    successes = {r: storage["success"][r][ds] for r in runs}
    any_run = runs[0]

    # All runs are faulty, return first (any will do)
    if all(successes[r] in [NA, UNK] for r in runs):
        return any_run

    # Return run with most successful frames
    succ_run, succ_val = max(successes.items(), key=lambda kv: kv[1])
    if list(successes.values()).count(succ_val) == 1:
        return succ_run

    # If there is a tie, select by lowest ATE
    runs = [r for r in runs if successes[r] == succ_val]
    ates = {r: storage["ate"][r][ds] for r in runs}
    ate_run, ate_val = min(ates.items(), key=lambda kv: kv[1][0])
    if list(ates.values()).count(ate_val) == 1:
        return ate_run

    # If still tie, select by lowest RTE
    runs = [r for r in runs if ates[r] == ate_val]
    rtes = {r: storage["rte"][r][ds] for r in runs}
    rte_run, rte_val = min(rtes.items(), key=lambda kv: kv[1][0])
    if list(rtes.values()).count(rte_val) == 1:
        return rte_run

    return any_run


def get_column_results(column, datasets):
    storage = load_storage([column["system"]], datasets)

    if "run" in column:  # A single deterministic run, used for every sequence
        run = column["run"]
        if run not in storage["completion"]:
            available = ", ".join(sorted(storage["completion"]))
            raise SystemExit(
                f"error: no run {run!r} in {BATCH_DIR}/*.{column['system']}.json (have: {available})"
            )
        selected_runs = {ds: run for ds in datasets}
    else:  # Several runs, the best one is picked per sequence
        runs = [f"{column['system']}.{i}" for i in column["runs"]]
        selected_runs = {ds: select_run(storage, ds, runs) for ds in datasets}

    return select_runs(storage, selected_runs)


# ---------------------------------------------------------------------------
# One table's worth of numbers
# ---------------------------------------------------------------------------


def sorted_and_p90(values, diverge_from, name):
    # Non-divergent values, sorted, plus their p90 (used as the colormap upper bound)
    finite = [x for x in values.reshape((-1,)) if np.isfinite(x)]
    kept = [x for x in sorted(finite) if x <= diverge_from]
    if not kept:  # Every run diverged (e.g. hard subsets like TO*), use the full range
        print(f"warning: all {name} values are above {diverge_from}, ignoring the divergence cutoff")
        kept = sorted(finite)
    if not kept:  # No usable value at all
        return [0.0, diverge_from], diverge_from
    return kept, kept[int(len(kept) * 0.9)]


def compute_table(datasets):
    """Everything one page needs: the four matrices, the row names and the colormap bounds."""
    results = [get_column_results(column, datasets) for column in COLUMNS]

    ates = np.ones((len(SYSTEMS), len(datasets))) * -1
    rtes = np.ones((len(SYSTEMS), len(datasets))) * -1
    completions = np.ones((len(SYSTEMS), len(datasets))) * -1
    successes = np.ones((len(SYSTEMS), len(datasets))) * -1

    for ds in datasets:
        for idx, _ in enumerate(SYSTEMS):
            ates[idx, datasets.index(ds)] = results[idx]["ates_avg"][ds]
            rtes[idx, datasets.index(ds)] = results[idx]["rtes_avg"][ds]
            completions[idx, datasets.index(ds)] = results[idx]["completion"][ds]
            successes[idx, datasets.index(ds)] = results[idx]["success"][ds]

    # Convert UNK and NA to neutral values
    ates[ates == NA] = np.inf
    rtes[rtes == NA] = np.inf
    completions[completions == NA] = -np.inf
    successes[successes == NA] = -np.inf
    ates[ates == UNK] = np.nan
    rtes[rtes == UNK] = np.nan
    completions[completions == UNK] = np.nan
    successes[successes == UNK] = np.nan

    # Convert from m to cm
    ates *= M_SCALER
    rtes *= M_SCALER
    # Convert to percentage
    completions *= 100
    successes *= 100

    # Get medians for each system, ignoring nan and infs
    ates_for_medians = np.nan_to_num(ates, nan=np.nan, posinf=np.nan)
    rtes_for_medians = np.nan_to_num(rtes, nan=np.nan, posinf=np.nan)
    completions_for_medians = np.nan_to_num(completions, nan=np.nan, neginf=np.nan)
    successes_for_medians = np.nan_to_num(successes, nan=np.nan, neginf=np.nan)
    ate_median = np.nanmedian(ates_for_medians, axis=1)
    rte_median = np.nanmedian(rtes_for_medians, axis=1)
    completion_median = np.nanmedian(completions_for_medians, axis=1)
    success_median = np.nanmedian(successes_for_medians, axis=1)

    ate_mean = np.nanmean(ates_for_medians, axis=1)
    rte_mean = np.nanmean(rtes_for_medians, axis=1)
    completion_mean = np.nanmean(completions_for_medians, axis=1)
    success_mean = np.nanmean(successes_for_medians, axis=1)

    # The colormap bounds come from the group's own values, like in benchmark.py
    ates_sorted, ate_p90 = sorted_and_p90(ates, ATE_DIVERGE_FROM, "ATE")
    rtes_sorted, rte_p90 = sorted_and_p90(rtes, RTE_DIVERGE_FROM, "RTE")

    ates = np.nan_to_num(ates, nan=U, posinf=N)
    rtes = np.nan_to_num(rtes, nan=U, posinf=N)
    completions = np.nan_to_num(completions, nan=-U, neginf=-N)
    successes = np.nan_to_num(successes, nan=-U, neginf=-N)

    # Add median row
    ates = np.hstack([ates, ate_median.reshape(len(SYSTEMS), 1)])
    rtes = np.hstack([rtes, rte_median.reshape(len(SYSTEMS), 1)])
    completions = np.hstack([completions, completion_median.reshape(len(SYSTEMS), 1)])
    successes = np.hstack([successes, success_median.reshape(len(SYSTEMS), 1)])

    # Add average row
    ates = np.hstack([ates, ate_mean.reshape(len(SYSTEMS), 1)])
    rtes = np.hstack([rtes, rte_mean.reshape(len(SYSTEMS), 1)])
    completions = np.hstack([completions, completion_mean.reshape(len(SYSTEMS), 1)])
    successes = np.hstack([successes, success_mean.reshape(len(SYSTEMS), 1)])

    return {
        "ates": ates,
        "rtes": rtes,
        "completions": completions,
        "successes": successes,
        "dataset_names": datasets + ["Median", "Average"],
        "ate_min": min(ates_sorted),
        "ate_max": ate_p90,
        "rte_min": min(rtes_sorted),
        "rte_max": rte_p90,
    }


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def create_cmap(colors=["#263238", "#ECEFF1"]):
    cmap = LinearSegmentedColormap.from_list("custom_rocket", colors)
    return cmap


ate_cmap = create_cmap(["#274d87", "#E3F2FD"])  # desaturated blue
rte_cmap = create_cmap(["#2d4c30", "#E8F5E9"])  # desaturated green
completions_cmap = create_cmap(["#ECEFF1", "#4e5e6f"])  # lighter dark grey
successes_cmap = create_cmap(["#ECEFF1", "#4e5e6f"])  # lighter dark grey


def plot_heatmaps(table, system_names):
    # Fixed height per row (in inches) so rows keep the same height regardless
    # of how many datasets are plotted.
    ROW_HEIGHT = 0.19  # inches per row
    FIG_WIDTH = 6.4638
    VERTICAL_MARGIN = 1.7  # inches reserved for titles, ticks and colorbar

    ates, rtes = table["ates"], table["rtes"]
    successes = table["successes"]
    dataset_names = table["dataset_names"]
    ate_min, ate_max = table["ate_min"], table["ate_max"]
    rte_min, rte_max = table["rte_min"], table["rte_max"]

    n_rows = len(dataset_names)
    fig_height = n_rows * ROW_HEIGHT + VERTICAL_MARGIN

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, fig_height), constrained_layout=True)

    cbar_kws = {"orientation": "horizontal", "location": "bottom"}
    tick_params_kws = {
        "axis": "x",
        "labelsize": 8,
        "labeltop": True,
        "top": True,
        "labelbottom": False,
        "bottom": False,
        "labelrotation": 45,
    }

    # Plotting Absolute Trajectory Error (ATE) heatmap
    annotate_ate = lambda x: ("×" if x == U else "×" if x == N else "∞" if x > ATE_DIVERGE_FROM else f"{x:.1f}")
    annotations_ate = np.vectorize(annotate_ate)(ates.T)
    sns.heatmap(
        ates.T,
        ax=axes[0],
        cmap=ate_cmap,
        annot=annotations_ate,
        fmt="",
        vmin=ate_min,
        vmax=ate_max,
        xticklabels=system_names,
        yticklabels=dataset_names,
        cbar_kws=cbar_kws | {"ticks": [ate_min, ate_max]},
    )
    axes[0].tick_params(**tick_params_kws)
    axes[0].set_title("ATE [cm] (SE3 aligned)", fontsize=10)

    # Plotting Relative Trajectory Error (RTE) heatmap
    annotate_rte = lambda x: ("×" if x == U else "×" if x == N else "∞" if x > RTE_DIVERGE_FROM else f"{x:.2f}")
    annotations_rte = np.vectorize(annotate_rte)(rtes.T)
    sns.heatmap(
        rtes.T,
        ax=axes[1],
        cmap=rte_cmap,
        annot=annotations_rte,
        fmt="",
        vmin=rte_min,
        vmax=rte_max,
        xticklabels=system_names,
        yticklabels=[],
        cbar_kws=cbar_kws | {"ticks": [rte_min, rte_max]},
    )
    axes[1].tick_params(**tick_params_kws)
    axes[1].set_title("RTE [cm] (Δ = 6 frames)", fontsize=10)

    # Plotting Success Percentage heatmap
    annotate_succ = lambda x: ("×" if x == -U else "×" if x == -N else "✓" if x > SUCCESS_FROM else f"{x:.0f}")
    annotations_succ = np.vectorize(annotate_succ)(successes.T)
    sns.heatmap(
        successes.T,
        ax=axes[2],
        cmap=successes_cmap,
        annot=annotations_succ,
        fmt="",
        vmin=0,
        vmax=100,
        xticklabels=system_names,
        yticklabels=[],
        cbar_kws=cbar_kws | {"ticks": [0, 100]},
    )
    axes[2].tick_params(**tick_params_kws)
    axes[2].set_title("Completed frames [%]", fontsize=10)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-p", "--prefixes", nargs="+", default=PREFIXES, metavar="PREFIX",
        help=f"dataset groups to plot, one page each (default: {' '.join(PREFIXES)})",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("all_benchmarks"),
        help="folder to write one <PREFIX>.pdf per group into (default: all_benchmarks)",
    )
    args = parser.parse_args()

    # Set font to CMU Serif
    plt.rcParams["font.family"] = ["CMU serif", "Sans-serif"]
    plt.rcParams["pdf.fonttype"] = 42  # see: http://phyletica.org/matplotlib-fonts/
    plt.rcParams["ps.fonttype"] = 42

    args.output.mkdir(parents=True, exist_ok=True)

    written = 0
    for prefix in args.prefixes:
        datasets = [ds for ds in DATASETS if ds.startswith(prefix)]
        if not datasets:
            print(f"warning: no dataset starts with {prefix!r}, skipping it")
            continue

        table = compute_table(datasets)
        fig = plot_heatmaps(table, SYSTEMS)

        out_file = args.output / f"{prefix}.pdf"
        fig.savefig(out_file)
        plt.close(fig)
        written += 1
        print(f"{prefix}: {len(datasets)} sequences -> {out_file}")

    print(f"\nWrote {written} PDFs to {args.output}/")


if __name__ == "__main__":
    main()
