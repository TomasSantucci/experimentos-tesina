#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from typing import Tuple, List

rcParams["font.family"] = "CMU Serif"
# CMU Serif's regular face reports weight 500, so the default "normal" (400)
# never matches; align every weight rcParam to 500 to avoid findfont warnings.
rcParams["font.weight"] = 500
rcParams["axes.labelweight"] = 500
rcParams["axes.titleweight"] = 500
rcParams["figure.titleweight"] = 500
rcParams["font.size"] = 16
rcParams["axes.labelsize"] = 16
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["legend.fontsize"] = 12
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

NUMBER_OF_NS_IN = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}

TIME_UNITS = NUMBER_OF_NS_IN.keys()

DEFAULT_TIME_UNITS = "ms"


def check_monotonic_rows(rows: np.ndarray) -> None:
    last_ts = 0
    for row in rows:
        ts = row[0]
        assert last_ts <= ts, f"Failed assertion {last_ts=} < {ts=}"
        last_ts = ts

def load_csv(csv_fn: Path, dtype=np.int64) -> np.ndarray:
    # print(csv_fn)
    try:
        data = np.genfromtxt(csv_fn, delimiter=",", comments="#", dtype=dtype, invalid_raise=True)
    except Exception as e:
        print(f"{csv_fn=}")
        print(e)
    check_monotonic_rows(data)
    return data

def load_csv_safer(csv_fn: Path, dtype=np.int64) -> Tuple[List[str], np.ndarray]:
    timing_data = load_csv(csv_fn, dtype)

    with open(csv_fn, "r", encoding="utf8") as f:
        first_line = next(f)
    assert first_line[0] == "#" and first_line[-1] == "\n", "first csv line should be a comment with column names"

    column_names = first_line[1:-1].split(",")
    assert len(column_names) == timing_data.shape[1], "number of column names differ from data columns"

    return column_names, timing_data

PGO_BUILD_COL = "lc_pgo_problem_built"
PGO_SOLVE_COL = "lc_pgo_problem_solved"
FRAME_TS_COL = 0  # First column is the frame timestamp


# Blue and red first, matching the rootba timings palette (fill / darker edge).
SERIES_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
                 "#00BCD4", "#E91E63", "#8BC34A", "#795548", "#607D8B"]
EDGE_COLORS   = ["#1565C0", "#BF360C", "#1B5E20", "#4A148C", "#E65100",
                 "#006064", "#880E4F", "#33691E", "#3E2723", "#263238"]


def load_pgo_data(csv_fn: Path, units: str = DEFAULT_TIME_UNITS):
    """Return (frame_tss_all, frame_tss_pgo, pgo_durations) for a CSV file."""
    column_names, timing_data = load_csv_safer(csv_fn)

    assert PGO_BUILD_COL in column_names, f"Column '{PGO_BUILD_COL}' not found in {column_names}"
    assert PGO_SOLVE_COL in column_names, f"Column '{PGO_SOLVE_COL}' not found in {column_names}"

    i_build = column_names.index(PGO_BUILD_COL)
    i_solve = column_names.index(PGO_SOLVE_COL)

    pgo_durations_ns = timing_data[:, i_solve] - timing_data[:, i_build]
    mask = pgo_durations_ns > 0

    frame_tss_all = (timing_data[:, FRAME_TS_COL] - timing_data[0, FRAME_TS_COL]) / NUMBER_OF_NS_IN["s"]
    frame_tss_pgo = frame_tss_all[mask]
    pgo_durations = pgo_durations_ns[mask] / NUMBER_OF_NS_IN[units]

    print(f"[{csv_fn}]")
    print(f"  Total frames:       {len(timing_data)}")
    print(f"  PGO frames:         {mask.sum()}")
    print(f"  PGO duration mean:  {pgo_durations.mean():.2f} {units}")
    print(f"  PGO duration std:   {pgo_durations.std():.2f} {units}")
    print(f"  PGO duration min:   {pgo_durations.min():.2f} {units}")
    print(f"  PGO duration max:   {pgo_durations.max():.2f} {units}")

    return frame_tss_all, frame_tss_pgo, pgo_durations


def plot_pgo_timing(
    csv_fns: list,
    units: str = DEFAULT_TIME_UNITS,
    save_path: Path = None,
    limit: float = None,
    title: str = None,
):
    dpi = 150
    fig, ax = plt.subplots(figsize=(2048 / dpi, 1024 / dpi), dpi=dpi)

    x_min, x_max = float("inf"), float("-inf")

    for idx, csv_fn in enumerate(csv_fns):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        edge  = EDGE_COLORS[idx % len(EDGE_COLORS)]
        frame_tss_all, frame_tss_pgo, pgo_durations = load_pgo_data(csv_fn, units)

        label = csv_fn.name if len(csv_fns) > 1 else f"PGO duration"
        label += f" (mean: {pgo_durations.mean():.2f} {units})"

        ax.plot(
            frame_tss_pgo,
            pgo_durations,
            marker="o",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor=edge,
            markeredgewidth=0.6,
            linewidth=1.2,
            color=color,
            alpha=0.9,
            solid_capstyle="round",
            label=label,
            zorder=3,
        )

        x_min = min(x_min, frame_tss_all[0])
        x_max = max(x_max, frame_tss_all[-1])

    if limit is not None:
        ax.axhline(
            limit,
            color="#FF5722",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=2,
            label=f"Approximate limit: {limit:.2f} {units}",
        )

    fig.suptitle(title if title is not None else "Pose Graph Optimization timing")
    legend = ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc",
                       fancybox=True, borderpad=0.8, labelspacing=0.6)
    legend.get_frame().set_linewidth(0.6)
    ax.yaxis.grid(True, color="#b0b0b0", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.tick_params(colors="#444444", length=3)
    ax.set_xlabel("Dataset time [s]")
    ax.set_ylabel(f"PGO duration [{units}]")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(0)

    fig.tight_layout(pad=2.0)

    if save_path is not None:
        fig.savefig(save_path)
        print(f"  Saved plot to: {save_path}")
    else:
        plt.show()


def parse_args():
    parser = ArgumentParser(
        description="Plot pose graph optimization (PGO) timing from a Monado timing CSV file.",
    )
    parser.add_argument(
        "timing_csvs",
        type=Path,
        nargs="+",
        help="Timing CSV file(s) generated from Monado",
    )
    parser.add_argument(
        "--save_plot",
        type=Path,
        default=None,
        help="Save the plot to this file instead of showing it",
    )
    parser.add_argument(
        "--units",
        type=str,
        default=DEFAULT_TIME_UNITS,
        choices=TIME_UNITS,
        help="Time units for the Y axis (default: ms)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Draw a horizontal dashed line at this value (in the chosen units)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_pgo_timing(args.timing_csvs, units=args.units, save_path=args.save_plot, limit=args.limit, title=args.title)


if __name__ == "__main__":
    main()
