"""
Generate a grouped box plot from total_times.json.
Each box represents the distribution of total_time across all datasets in a group,
grouped by dataset collection (tumvi / euroc / msd).

Usage:
    python plot_timing.py --dataset tumvi [--input total_times.json] [--output timing_plot.pdf]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from pathlib import Path

rcParams["font.family"] = "CMU Serif"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Dataset tree
# ---------------------------------------------------------------------------
DATASET_TREE = {
    "tumvi": {
        "corridor":   ["TC1","TC2","TC3","TC4","TC5"],
        "magistrale": ["TM1","TM2","TM3","TM4","TM5","TM6"],
        "outdoors":   ["TO1","TO2","TO3","TO4","TO5","TO6","TO7","TO8"],
        "room":       ["TR1","TR2","TR3","TR4","TR5","TR6"],
        "slides":     ["TS1","TS2","TS3"],
    },
    "euroc": {
        "MH": ["EMH01","EMH02","EMH03","EMH04","EMH05"],
        "V1": ["EV101","EV102","EV103"],
        "V2": ["EV201","EV202","EV203"],
    },
    "msd": {
        "msdmi": ["MIO01","MIO02","MIO03","MIO04","MIO05","MIO06","MIO07","MIO08",
                  "MIO09","MIO10","MIO11","MIO12","MIO13","MIO14","MIO15","MIO16",
                  "MIPB01","MIPB02","MIPB03","MIPB04","MIPB05","MIPB06","MIPB07","MIPB08",
                  "MIPP01","MIPP02","MIPP03","MIPP04","MIPP05","MIPP06",
                  "MIPT01","MIPT02","MIPT03"],
        "msdgo": ["MGO01","MGO02","MGO03","MGO04","MGO05","MGO06","MGO07","MGO08",
                  "MGO09","MGO10","MGO11","MGO12","MGO13","MGO14","MGO15"],
        "msmoo": ["MOO01","MOO02","MOO03","MOO04","MOO05","MOO06","MOO07","MOO08",
                  "MOO09","MOO10","MOO11","MOO12","MOO13","MOO14","MOO15","MOO16"],
    },
}

VALID_GROUPS = list(DATASET_TREE.keys())

BAR_COLORS  = ["#FF5722", "#2196F3", "#4CAF50", "#9C27B0", "#FF9800",
               "#00BCD4", "#E91E63", "#8BC34A", "#795548", "#607D8B"]
EDGE_COLORS = ["#BF360C", "#1565C0", "#1B5E20", "#4A148C", "#E65100",
               "#006064", "#880E4F", "#33691E", "#3E2723", "#263238"]


def collect_times(run_data: dict, datasets: list[str]) -> list[float]:
    return [run_data[d]["total_time"] for d in datasets if d in run_data]


def print_median_stats(data: dict, ds_group: str, flat: bool = False) -> None:
    """Print median total_time per group and per run, plus difference/ratio between runs."""
    groups = build_groups(ds_group, flat)
    runs   = list(data.keys())

    col_w = 14
    header_parts = [f"{'group':<12}"] + [f"{r:>{col_w}}" for r in runs]
    if len(runs) == 2:
        header_parts += [f"{'abs diff':>{col_w}}", f"{'rel diff (%)':>{col_w}}"]
    print("\n" + "=" * (12 + col_w * len(header_parts) - 12))
    print(f"Median total_time  [{ds_group.upper()}]")
    print("-" * (12 + col_w * (len(runs) + (2 if len(runs) == 2 else 0))))
    print("".join(header_parts))
    print("-" * (12 + col_w * (len(runs) + (2 if len(runs) == 2 else 0))))

    all_medians: dict[str, list[float]] = {r: [] for r in runs}

    for _, group_name, datasets in groups:
        medians = {}
        row = f"{group_name:<12}"
        for run in runs:
            vals = collect_times(data[run], datasets)
            med  = float(np.median(vals)) if vals else float("nan")
            medians[run] = med
            all_medians[run].append(med)
            row += f"{med:>{col_w}.3f}"
        if len(runs) == 2:
            m0, m1 = medians[runs[0]], medians[runs[1]]
            abs_diff = m1 - m0
            rel_diff = (abs_diff / m0 * 100) if m0 != 0 else float("nan")
            row += f"{abs_diff:>{col_w}.3f}{rel_diff:>{col_w}.1f}"
        print(row)

    if flat:
        print("=" * (12 + col_w * (len(runs) + (2 if len(runs) == 2 else 0))) + "\n")
        return

    # Overall row (median of group medians)
    print("-" * (12 + col_w * (len(runs) + (2 if len(runs) == 2 else 0))))
    overall_row = f"{'OVERALL':<12}"
    overall_medians = {}
    for run in runs:
        vals = [v for v in all_medians[run] if not np.isnan(v)]
        med  = float(np.median(vals)) if vals else float("nan")
        overall_medians[run] = med
        overall_row += f"{med:>{col_w}.3f}"
    if len(runs) == 2:
        m0, m1 = overall_medians[runs[0]], overall_medians[runs[1]]
        abs_diff = m1 - m0
        rel_diff = (abs_diff / m0 * 100) if m0 != 0 else float("nan")
        overall_row += f"{abs_diff:>{col_w}.3f}{rel_diff:>{col_w}.1f}"
    print(overall_row)
    print("=" * (12 + col_w * (len(runs) + (2 if len(runs) == 2 else 0))) + "\n")


def build_groups(ds_group: str, flat: bool = False) -> list[tuple[str, str, list[str]]]:
    if flat:
        all_datasets = [
            d for datasets in DATASET_TREE[ds_group].values() for d in datasets
        ]
        return [(ds_group, ds_group, all_datasets)]
    return [
        (ds_group, group_name, datasets)
        for group_name, datasets in DATASET_TREE[ds_group].items()
    ]


def plot(data: dict, output: str | None, ds_group: str, width: int, height: int, dpi: int, flat: bool = False):
    print_median_stats(data, ds_group, flat)
    groups       = build_groups(ds_group, flat)
    group_labels = [g[1] for g in groups]
    runs         = list(data.keys())
    n_groups     = len(groups)
    n_runs       = len(runs)

    box_width = 0.7 / n_runs
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    for i, run in enumerate(runs):
        bar_color  = BAR_COLORS[i % len(BAR_COLORS)]
        edge_color = EDGE_COLORS[i % len(EDGE_COLORS)]
        offsets    = x - 0.35 + box_width * (i + 0.5)

        all_vals = []
        for _, _, datasets in groups:
            all_vals.append(collect_times(data[run], datasets))

        bp = ax.boxplot(
            all_vals,
            positions=offsets,
            widths=box_width * 0.85,
            patch_artist=True,
            manage_ticks=False,
            zorder=3,
            boxprops=dict(facecolor=bar_color, alpha=0.6, linewidth=1.2, edgecolor=edge_color),
            medianprops=dict(color=edge_color, linewidth=2.0),
            whiskerprops=dict(color=edge_color, linewidth=1.2, linestyle="--"),
            capprops=dict(color=edge_color, linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor=bar_color, markeredgecolor=edge_color,
                            markersize=4, alpha=0.6, linewidth=0.8),
        )

    title_suffix = "(all sequences pooled)" if flat else "dataset groups"
    ax.set_title(f"Total Time — {ds_group.upper()} {title_suffix}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=35, ha="right", fontsize=12)
    ax.set_ylabel("total time (s)", fontsize=13)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(visible=True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Patch(facecolor=BAR_COLORS[i % len(BAR_COLORS)],
              edgecolor=EDGE_COLORS[i % len(EDGE_COLORS)],
              alpha=0.6, label=run)
        for i, run in enumerate(runs)
    ]
    ax.legend(handles=legend_handles, title="run", fontsize=11,
              title_fontsize=11, loc="upper right", framealpha=0.8)

    fig.tight_layout(pad=2.5)

    if output is not None:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="total_times.json", help="Input JSON file")
    parser.add_argument("--output", default=None,               help="Output plot file (pdf/png/svg)")
    parser.add_argument("--width",  type=int, default=1280,     help="Figure width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=560,      help="Figure height in pixels (default: 560)")
    parser.add_argument("--dpi",    type=int, default=150,      help="DPI (default: 150)")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=VALID_GROUPS,
        help=f"Dataset group to plot. One of: {', '.join(VALID_GROUPS)}",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Pool all subgroups into a single box instead of splitting by subgroup",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    plot(data, args.output, args.dataset, args.width, args.height, args.dpi, args.flat)


if __name__ == "__main__":
    main()
