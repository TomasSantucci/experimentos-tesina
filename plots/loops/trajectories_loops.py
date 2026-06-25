#!/usr/bin/env python3
"""
Plot a EuRoC-format trajectory in 3D (X, Y, time) with loop detection
and closure events marked as colored lines between the two involved poses.

Single-dataset usage:
    python trajectories_loops.py \
        --traj  tracking.csv \
        --log   slam.log \
        [--out  output.png]   # optional; if omitted an interactive window opens

Batch usage (iterate over all dataset sub-directories):
    python trajectories_loops.py \
        --input-dir /path/to/run_dir \
        --out-dir   /path/to/output_pdfs   # optional; defaults to <input-dir>/plots
    
    Each sub-directory must contain tracking.csv and output.log.
    One PDF per dataset is saved as <out-dir>/<dataset_name>.pdf.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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


# ── helpers ──────────────────────────────────────────────────────────────────

def load_trajectory(path: str) -> pd.DataFrame:
    """Read a EuRoC CSV trajectory file."""
    df = pd.read_csv(
        path,
        comment="#",
        header=0,
        names=[
            "t_ns", "x", "y", "z",
            "qw", "qx", "qy", "qz",
        ],
    )
    df["t_ns"] = df["t_ns"].astype(np.int64)
    return df


def parse_loops(log_text: str) -> list[dict]:
    """
    Extract loop events from the log.

    Each event is a dict with keys:
        ts_cur   : int   current keyframe timestamp [ns]
        ts_cand  : int   candidate keyframe timestamp [ns]
        accepted : bool  True if ACCEPTED, False if REJECTED
    """
    # Pattern: [LC]  Loop <ts1> <-> <ts2>   ...then on a later line...  -> ACCEPTED / REJECTED
    loop_block = re.compile(
        r"\[LC\]\s+Loop\s+(\d+)\s+<->\s+(\d+).*?"
        r"(ACCEPTED|REJECTED)",
        re.DOTALL,
    )
    events = []
    for m in loop_block.finditer(log_text):
        events.append(
            {
                "ts_cur": int(m.group(1)),
                "ts_cand": int(m.group(2)),
                "accepted": m.group(3) == "ACCEPTED",
            }
        )
    return events


def nearest_pose(df: pd.DataFrame, ts_ns: int):
    """Return (x, y, t_sec) for the trajectory row closest to ts_ns."""
    idx = (df["t_ns"] - ts_ns).abs().argmin()
    row = df.iloc[idx]
    t_sec = (row["t_ns"] - df["t_ns"].iloc[0]) * 1e-9
    return float(row["x"]), float(row["y"]), t_sec


# ── plot ─────────────────────────────────────────────────────────────────────

def plot_trajectory(traj_path: str, log_path: str, title: str = None) -> plt.Figure:
    """Build and return a figure for one dataset."""
    traj = load_trajectory(traj_path)
    log_text = Path(log_path).read_text(errors="replace")
    loops = parse_loops(log_text)

    if not loops:
        print(f"  No loop events found in {log_path}", file=sys.stderr)

    # Time axis: seconds elapsed from first pose
    t0 = traj["t_ns"].iloc[0]
    traj["t_sec"] = (traj["t_ns"] - t0) * 1e-9

    # ── figure ──
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── styling: soft panes, light gridlines, gentle viewing angle ──
    ax.view_init(elev=22, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.985, 0.985, 0.99, 1.0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        axis.pane.set_linewidth(0.6)
        axis._axinfo["grid"].update(color=(0.86, 0.86, 0.88, 1.0), linewidth=0.6)
        axis.set_tick_params(colors="#444444", length=3, pad=4)

    # Trajectory line
    ax.plot(
        traj["x"].values,
        traj["y"].values,
        traj["t_sec"].values,
        color="#2b2b2b",
        linewidth=1.4,
        solid_capstyle="round",
        alpha=0.9,
        label="Trajectory",
        zorder=2,
    )

    # Loop lines
    n_accepted = n_rejected = 0
    for ev in loops:
        x1, y1, t1 = nearest_pose(traj, ev["ts_cur"])
        x2, y2, t2 = nearest_pose(traj, ev["ts_cand"])

        if ev["accepted"]:
            color, lw, al, zo = "#1a9850", 2.0, 0.9, 4
            n_accepted += 1
        else:
            color, lw, al, zo = "#d73027", 1.1, 0.5, 3
            n_rejected += 1

        ax.plot(
            [x1, x2], [y1, y2], [t1, t2],
            color=color,
            linewidth=lw,
            alpha=al,
            solid_capstyle="round",
            zorder=zo,
        )

    # Legend proxies for loop lines
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="#d73027", linewidth=1.6,
               label=f"Loop detected – rejected ({n_rejected})"),
        Line2D([0], [0], color="#1a9850", linewidth=2.0,
               label=f"Loop closed – accepted ({n_accepted})"),
    ]

    ax.set_xlabel("x (m)", labelpad=10)
    ax.set_ylabel("y (m)", labelpad=10)
    ax.set_zlabel("time (s)", labelpad=10)

    legend = ax.legend(
        handles=handles,
        loc="upper left",
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=True,
        borderpad=0.8,
        labelspacing=0.6,
        handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Single-dataset mode
    single = parser.add_argument_group("single-dataset mode")
    single.add_argument("--traj", default=None, help="EuRoC CSV trajectory file")
    single.add_argument("--log",  default=None, help="SLAM log file")
    single.add_argument("--out",  default=None, help="Output image path (optional)")
    single.add_argument(
        "--no-interactive",
        action="store_true",
        help="Suppress the interactive window even when --out is not given",
    )

    # Batch mode
    batch = parser.add_argument_group("batch mode")
    batch.add_argument(
        "--input-dir", default=None,
        help="Directory whose sub-directories are individual datasets "
             "(each must contain tracking.csv and output.log)",
    )
    batch.add_argument(
        "--out-dir", default=None,
        help="Directory where PDFs are saved (default: <input-dir>/plots)",
    )

    args = parser.parse_args()

    # ── batch mode ──────────────────────────────────────────────────────────
    if args.input_dir is not None:
        matplotlib.use("Agg")   # no display needed
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            sys.exit(f"ERROR: --input-dir '{input_dir}' is not a directory.")

        out_dir = Path(args.out_dir) if args.out_dir else input_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        datasets = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir()
            and (d / "tracking.csv").exists()
            and (d / "output.log").exists()
        )

        if not datasets:
            sys.exit(f"ERROR: no datasets found in '{input_dir}' "
                     "(sub-dirs must contain tracking.csv and output.log).")

        print(f"Found {len(datasets)} dataset(s) in '{input_dir}'.")
        for ds in datasets:
            out_path = out_dir / f"{ds.name}.pdf"
            print(f"  Processing {ds.name} …", end=" ", flush=True)
            try:
                fig = plot_trajectory(
                    traj_path=str(ds / "tracking.csv"),
                    log_path=str(ds / "output.log"),
                    title=f"{ds.name} – trajectory with loop detections (X / Y / time)",
                )
                fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.3)
                plt.close(fig)
                print(f"saved → {out_path}")
            except Exception as exc:
                print(f"FAILED ({exc})", file=sys.stderr)

        print("Done.")
        return

    # ── single-dataset mode ─────────────────────────────────────────────────
    if args.traj is None or args.log is None:
        parser.error("Provide either --input-dir (batch) or both --traj and --log (single).")

    fig = plot_trajectory(args.traj, args.log)

    if args.out:
        out_path = Path(args.out)
        fmt = out_path.suffix.lstrip(".") or "pdf"
        fig.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.3)
        print(f"Saved → {out_path}")

    if not args.out or not args.no_interactive:
        plt.show()


if __name__ == "__main__":
    main()
