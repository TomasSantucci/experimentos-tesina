# NOTE: The timing numbers coming from this directory come from running on a
# Intel Core Ultra 9 285H (Thinkpad P16S laptop)

import argparse
from itertools import cycle
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

CONFIGS: dict[str, dict] = {
    "msdmg": dict(fps=30, zoom=33.3),
    "msdmo": dict(fps=30, zoom=33.3),
    "msdmi": dict(fps=54, zoom=18.5),
    "euroc":  dict(fps=20, zoom=50.0),
    "tum":    dict(fps=30, zoom=33.3),
}

SYSTEMS: dict[str, str] = {
    "Basalt-VIO":     "basalt",
    "Basalt-LC": "basalt_lc",
    "Basalt-LC-CQ": "basalt_lc_cq",
    "OKVIS2":     "okvis2",
    "ORB-SLAM3":  "orbslam3",
    "DM-VIO":     "dmvio",
    "SnakeSLAM":  "snakeslam",
}

COLORS: list[str] = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FFC107",  # amber
    "#E91E63",  # pink
    "#673AB7",  # deeppurple
    "#00BCD4",  # cyan
    "#CDDC39",  # lime
    "#FF5722",  # deeporange
    "#009688",  # teal
]

DEFAULT_SYSTEMS = ["Basalt", "OKVIS2", "ORB-SLAM3"]


def _parse_csv(path: Path) -> tuple[list[int], list[int]]:
    """Return (frame_timestamps_ns, wall_timestamps_ns) from a timing CSV."""
    rows = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    frames = [int(r.split(",")[0]) for r in rows]
    walls  = [int(r.split(",")[1]) for r in rows]
    return frames, walls


def load_system_timings(
    system_dir: str, dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load in/out CSVs for one system and return (frame_ts_ns, latency_ms)."""
    base = Path(system_dir) / dataset
    in_frames,  in_walls  = _parse_csv(base / "in.csv")
    out_frames, out_walls = _parse_csv(base / "out.csv")
    if in_frames != out_frames:
        raise ValueError(f"Frame mismatch in {base}")
    frame_ts_ns = np.array(in_frames)
    latency_ms  = (np.array(out_walls) - np.array(in_walls)) / 1e6
    return frame_ts_ns, latency_ms


def infer_dataset_meta(system_dir: str, dataset: str) -> tuple[int, int]:
    """Return (frame_count, first_timestamp_ns) from the first system's in.csv."""
    frames, _ = _parse_csv(Path(system_dir) / dataset / "in.csv")
    return len(frames), frames[0]


def print_statistics(name: str, latency_ms: np.ndarray) -> None:
    print(f"[{name}]")
    print(f"  mean   = {latency_ms.mean():.2f} ms")
    print(f"  median = {median(latency_ms):.2f} ms")
    print(f"  std    = {latency_ms.std():.2f} ms")
    print(f"  min    = {latency_ms.min():.2f} ms")
    print(f"  max    = {latency_ms.max():.2f} ms")


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_system(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    name: str,
    color: str,
    zorder: int,
    window: int,
    show_mean: bool = False,
) -> None:
    """Plot raw timings (faint) and their moving average for one system."""
    ax.plot(xs, ys, alpha=0.15, color=color, label="", zorder=zorder)
    ma = moving_average(ys, window)
    if show_mean:
        name += f" ({ys.mean():.1f} ± {ys.std():.1f} ms)"
    ax.plot(
        xs[: len(ma)], ma,
        color=color, alpha=0.7, label=name, zorder=zorder - 10,
    )


def build_figure(args: argparse.Namespace) -> plt.Figure:
    cfg = CONFIGS[args.config]
    first_system_dir = SYSTEMS[args.systems[0]]
    frame_count, first_ts = infer_dataset_meta(first_system_dir, args.dataset)

    maxx     = frame_count / cfg["fps"]
    realtime = args.realtime if args.realtime is not None else cfg["zoom"]
    title    = args.title or f"Frame timings on {args.dataset} dataset"

    fig, ax = plt.subplots(
        figsize=(args.width / args.dpi, args.height / args.dpi),
        dpi=args.dpi,
    )
    fig.tight_layout(pad=1.75)
    ax.set_xlim(0, maxx)
    ax.set_ylim(0, 100)
    ax.axhline(y=realtime, color="#F44336", linestyle="--", alpha=1.0,
               label=f"Real-time ({realtime:.0f} ms)")

    for name, color, zorder in zip(args.systems, cycle(COLORS), range(3, 3 + len(args.systems))):
        frame_ts, latency_ms = load_system_timings(SYSTEMS[name], args.dataset)
        xs = (frame_ts - first_ts) / 1e9
        plot_system(ax, xs, latency_ms, name, color, zorder, args.window, args.show_mean)
        print_statistics(name, latency_ms)

    ax.set_xlabel("Dataset time [s]", fontsize=14)
    ax.set_ylabel("Frame time [ms]", labelpad=0, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="upper center", ncol=len(args.systems), columnspacing=1, handlelength=1.5)
    ax.grid(visible=True, alpha=0.3)

    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-frame timings for VIO/SLAM systems."
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Dataset name — must match the subdirectory inside each system folder (e.g. MGO04)",
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        choices=list(CONFIGS.keys()),
        help=f"Dataset family config (sets fps and real-time budget). Choices: {list(CONFIGS.keys())}",
    )
    parser.add_argument(
        "--systems", "-s",
        nargs="+",
        default=DEFAULT_SYSTEMS,
        choices=list(SYSTEMS.keys()),
        metavar="SYSTEM",
        help=f"Systems to include. Available: {list(SYSTEMS.keys())}. Default: {DEFAULT_SYSTEMS}",
    )
    parser.add_argument("--maxy",   type=float, default=200.0, help="Max y-axis value in ms (default: 200)")
    parser.add_argument("--window", "-w", type=int, default=60, help="Moving-average window in frames (default: 60)")
    parser.add_argument("--width",  type=int, default=1024, help="Figure width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=640,  help="Figure height in pixels (default: 640)")
    parser.add_argument("--dpi",    type=int, default=150,  help="DPI (default: 150)")
    parser.add_argument("--title",  type=str, default=None, help="Plot title (default: auto-generated)")
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Save figure to this file instead of displaying it (e.g. plot.pdf)",
    )
    parser.add_argument(
        "--realtime", "-r",
        type=float, default=None,
        help="Y value (ms) for the real-time budget line (default: from config)",
    )
    parser.add_argument(
        "--show-mean", "-m",
        action="store_true",
        default=False,
        help="Overlay a horizontal dotted line for the mean latency of each system and add it to the legend",
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
