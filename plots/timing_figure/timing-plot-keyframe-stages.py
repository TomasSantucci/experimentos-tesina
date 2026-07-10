# NOTE: The timing numbers coming from this directory come from running on a
# Intel Core Ultra 9 285H (Thinkpad P16S laptop)
#
# Unlike timing-plot.py (which reads per-system in.csv/out.csv pairs), this
# script takes a single full per-frame timings CSV (e.g. culling.timings.csv)
# and plots the end-to-end frame latency straight from it. The latency of each
# frame is measured from `frontend_frames_received` to `consumer_state_received`.

import argparse
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "CMU Serif"
# Use type3 fonts to pass the ieee ras papercept pdf check
# see: http://phyletica.org/matplotlib-fonts/
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

# Dataset time column (ns).
FRAME_TS_COL = "frames_original_timestamp"

# Ordered boundary columns (by header name). Each consecutive pair defines one
# stacked stage; the latency of stage i is BOUNDARIES[i+1] - BOUNDARIES[i].
BOUNDARIES = [
    "frontend_frames_received",
    "frontend_keypoints_pushed",
    "backend_marginalization_ended",
    "map_stamp_saved",
    "lc_loop_detection_finished",
    "consumer_state_received",
]

# One label + color per stage (len == len(BOUNDARIES) - 1). Avoid the red used
# for the real-time line.
STAGE_LABELS = [
    "Front-end",
    "VIO",
    "Map update",
    "Loop detection",
    "Loop optimization",
]
# Palette options for the stages. The last entry is the "Loop optimization"
# stage; uncomment the one you like best (comment out the rest).
#STAGE_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#80DEEA"]  # light cyan
STAGE_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#80CBC4"]  # light teal

# Which stages *can* run in the background (they were run sequentially in this
# experiment). These are drawn with a hatch pattern and marked with a "*" in the
# legend, explained by a footnote.
STAGE_ASYNC = [False, False, True, True, True]
ASYNC_FOOTNOTE = "* Stage can run in the background (run sequentially here)."


def load_timings(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load a full timings CSV and return (frame_ts_ns, [stage_ms, ...])."""
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    header = lines[0].lstrip("#").split(",")
    idx = {name: i for i, name in enumerate(header)}
    for col in (FRAME_TS_COL, *BOUNDARIES):
        if col not in idx:
            raise ValueError(f"Column '{col}' not found in {path}")

    data = np.array([[int(x) for x in r.split(",")] for r in lines[1:]])
    frame_ts_ns = data[:, idx[FRAME_TS_COL]]
    stages_ms = [
        (data[:, idx[hi]] - data[:, idx[lo]]) / 1e6
        for lo, hi in zip(BOUNDARIES, BOUNDARIES[1:])
    ]
    return frame_ts_ns, stages_ms


def print_statistics(name: str, latency_ms: np.ndarray) -> None:
    print(f"[{name}]")
    print(f"  mean   = {latency_ms.mean():.2f} ms")
    print(f"  median = {median(latency_ms):.2f} ms")
    print(f"  std    = {latency_ms.std():.2f} ms")
    print(f"  min    = {latency_ms.min():.2f} ms")
    print(f"  max    = {latency_ms.max():.2f} ms")


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_timings(
    ax: plt.Axes,
    xs: np.ndarray,
    stages: list[np.ndarray],
    window: int,
    show_mean: bool = False,
) -> None:
    """Plot the timing stages as stacked filled bands.

    Every stage is smoothed with a moving average and stacked on top of the
    previous one (like a stacked barplot), except the LAST stage, which is drawn
    from the raw per-frame data. The bottom stage is filled down to zero.
    """
    # All series must share a common length. The moving average (mode="valid")
    # trims to len - window + 1, so raw series are sliced to match.
    length = len(stages[0]) - window + 1
    x = xs[:length]

    baseline = np.zeros(length)
    for i, stage in enumerate(stages):
        is_last = i == len(stages) - 1
        # Last stage uses raw per-frame data; the rest use the moving average.
        series = stage[:length] if is_last else moving_average(stage, window)
        top = baseline + series

        is_async = STAGE_ASYNC[i]
        label = STAGE_LABELS[i]
        if is_async:
            label += "*"
        if show_mean:
            label += f" ({stage.mean():.1f} ± {stage.std():.1f} ms)"

        color = STAGE_COLORS[i]
        # Async stages get a hatch pattern so they read as a separate group.
        ax.fill_between(x, baseline, top, facecolor=color, alpha=0.35,
                        hatch="///" if is_async else None,
                        edgecolor=color if is_async else "none",
                        label=label, zorder=1 + i)
        ax.plot(x, top, color=color, alpha=0.9,
                linewidth=0.6 if is_last else 1.2, zorder=100 + i)
        baseline = top


def build_figure(args: argparse.Namespace) -> plt.Figure:
    frame_ts, stages_ms = load_timings(args.input)
    xs = (frame_ts - frame_ts[0]) / 1e9

    # X extent comes from the actual dataset-time span: frames may be decimated
    # (culling/keyframing), so len(frames) / fps would underestimate it.
    maxx  = float(xs[-1])
    title = args.title or f"Keyframe Timings on MGO12"

    fig, ax = plt.subplots(
        figsize=(args.width / args.dpi, args.height / args.dpi),
        dpi=args.dpi,
    )
    fig.tight_layout(pad=1.75)
    ax.set_xlim(0, maxx)
    ax.set_ylim(0, args.maxy)
    #ax.axhline(y=args.realtime, color="#F44336", linestyle="--", alpha=1.0,
    #           label=f"Real-time ({args.realtime:.0f} ms)")

    name = args.label or args.input.stem
    plot_timings(ax, xs, stages_ms, args.window, args.show_mean)
    for label, stage in zip(STAGE_LABELS, stages_ms):
        print_statistics(f"{name} ({label})", stage)
    print_statistics(f"{name} (total)", sum(stages_ms))

    ax.set_xlabel("Dataset time [s]", fontsize=14)
    ax.set_ylabel("Processing duration [ms]", labelpad=0, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="upper left", columnspacing=1, handlelength=1.5)
    ax.grid(visible=True, alpha=0.3)

    if any(STAGE_ASYNC):
        # Reserve room at the bottom so the footnote stays inside the canvas
        # (works with plt.show(), not only with bbox_inches="tight" on save).
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.02, 0.015, ASYNC_FOOTNOTE, ha="left", va="bottom",
                 fontsize=10, style="italic")

    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-frame end-to-end timings from a full timings CSV."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a full per-frame timings CSV (e.g. culling.timings.csv)",
    )
    parser.add_argument("--realtime", "-r", type=float, default=33.3, help="Real-time budget line in ms (default: 33.3)")
    parser.add_argument("--maxy",     type=float, default=100.0, help="Max y-axis value in ms (default: 100)")
    parser.add_argument("--window",   "-w", type=int, default=20, help="Moving-average window in frames (default: 60)")
    parser.add_argument("--width",    type=int, default=1024, help="Figure width in pixels (default: 1024)")
    parser.add_argument("--height",   type=int, default=640,  help="Figure height in pixels (default: 640)")
    parser.add_argument("--dpi",      type=int, default=150,  help="DPI (default: 150)")
    parser.add_argument("--title",    type=str, default=None, help="Plot title (default: auto-generated)")
    parser.add_argument("--label",    type=str, default=None, help="Legend label for the series (default: file stem)")
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Save figure to this file instead of displaying it (e.g. plot.pdf)",
    )
    parser.add_argument(
        "--show-mean", "-m",
        action="store_true",
        default=False,
        help="Add the mean ± std latency to the legend label",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fig = build_figure(args)
    if args.output:
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
