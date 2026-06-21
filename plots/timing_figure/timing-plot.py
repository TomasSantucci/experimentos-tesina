# NOTE: The timing numbers coming from this directory come from running on a
# Intel Core Ultra 7 258V (Asus Zenbook S24 laptop), they are not to be found
# in the timing results from the root directory. Read the README.md file in that
# directory for more information about timing info.

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import cycle
from statistics import median

rcParams["font.family"] = "CMU Serif"
rcParams["font.family"] = "CMU Serif"
# Use type3 fonts to pass the ieee ras papercept pdf check
# see: http://phyletica.org/matplotlib-fonts/
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

DATASET_FRAME_COUNT = 5404  # MOO02
DATASET_FIRST_TS = 7669821982615  # MOO02
DPI = 150
IMG_WIDTH = 1024
IMG_HEIGHT = 640
MAXY = 110
MAXX = DATASET_FRAME_COUNT * 1 / 30
Z = 33.33333  # Zoom from 0 to Z
M = 4  # Multiplier for the zoomed area
WINDOW_SIZE = 60
yticks = sorted(list(np.linspace(0, MAXY, MAXY // 10 + 1, endpoint=True).astype(int)) + [Z])
yticks_labels = yticks.copy()
# yticks_labels[yticks_labels.index(Z)] = f"{Z:.0f}"
yticks_labels[yticks_labels.index(Z)] = f"{Z:.0f}"

COLORS = [
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


def statistics(name, arr: np.ndarray):
    print(f"[{name}]")
    print(f"mean={arr.mean():.2f}ms")
    print(f"median={median(arr):.2f}ms")
    print(f"std={arr.std():.2f}ms")
    print(f"min={arr.min():.2f}ms")
    print(f"max={arr.max():.2f}ms")
    print()


def moving_average(data, window_size=WINDOW_SIZE):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


ins = ["basalt/in.csv", "okvis2/in.csv", "orbslam3/in.csv", "dmvio/in.csv", "snakeslam/in.csv"]
outs = ["basalt/out.csv", "okvis2/out.csv", "orbslam3/out.csv", "dmvio/out.csv", "snakeslam/out.csv"]
names = ["Basalt", "OKVIS2", "ORB-SLAM3", "DM-VIO", "SnakeSLAM"]
zorder = [5, 4, 5, 1, 1]
totals = [24.38, 146.20, 209.63, 95.07, 40.96]  # Basalt and snakeslame totals are from parallel runs
# It is unfair for snakeslam and ORB-SLAM3 to show 40.96, because i am not showing ATE for global ba

fig, ax = plt.subplots(figsize=(IMG_WIDTH / DPI, IMG_HEIGHT / DPI), dpi=DPI)
fig.tight_layout(pad=1.75)
plt.ylim(0, MAXY)
plt.xlim(0, MAXX)


# Set y scale to be linear from 0-Z, and then 2x from Z-MAXY
# plt.fill_between(xs, 0, Z, color="gray", alpha=0.1)
# plt.fill_between([0, MAXX], Z, MAXY, color="#ECEFF1", alpha=1.0)
plt.axhline(y=Z, color="#F44336", linestyle="--", alpha=1.0)
f_fwd = lambda ys: np.vectorize(lambda x: x if x < Z else Z + (x - Z) / M, otypes=[ys.dtype])(ys)
f_bwd = lambda ys: np.vectorize(lambda x: x if x < Z else Z + (x - Z) * M, otypes=[ys.dtype])(ys)
plt.yscale("function", functions=(f_fwd, f_bwd))
plt.yticks(yticks, yticks_labels)

colors = cycle(COLORS)
for name, in_fn, out_fn, color, zorder in zip(names, ins, outs, colors, zorder):
    incsv = open(in_fn).readlines()
    outcsv = open(out_fn).readlines()
    assert len(incsv) == len(outcsv)

    in_frames = [int(l.split(",")[0].strip()) for l in incsv if not l.startswith("#")]
    out_frames = [int(l.split(",")[0].strip()) for l in outcsv if not l.startswith("#")]
    assert in_frames == out_frames, name
    xs = (np.array(in_frames) - DATASET_FIRST_TS) / 1e9

    in_tss = [int(l.split(",")[1].strip()) for l in incsv if not l.startswith("#")]
    out_tss = [int(l.split(",")[1].strip()) for l in outcsv if not l.startswith("#")]
    diff_tss = (np.array(out_tss) - np.array(in_tss)) / 1e6
    ys = diff_tss

    plot0 = plt.plot(xs, ys, alpha=0.15, color=color, label="", zorder=zorder)
    plt.plot(xs[: len(moving_average(ys))], moving_average(ys), color=plot0[0].get_color(), alpha=0.7, label=name, zorder=zorder - 10)

    # plt.plot(xs[: len(moving_average(ys))], moving_average(ys), color=color, alpha=0.7, label=name, zorder=zorder)

    # plot0 = plt.plot(xs, ys, alpha=0.5, color=color, label="", zorder=zorder)
    # plt.plot(xs[: len(moving_average(ys))], moving_average(ys), color=plot0[0].get_color(), alpha=0.7, label=name, zorder=zorder)

    statistics(name, diff_tss)

plt.xlabel("Dataset time [s]", fontsize=14)
plt.ylabel("Frame time [ms]", labelpad=0, fontsize=14)
plt.title("Frame timings on MOO02 dataset", fontsize=16)

plt.legend(loc="upper center", ncol=len(names), columnspacing=1, handlelength=1.5)  # fontsize="small"?
plt.grid(visible=True, alpha=0.3)
plt.show()

"""
// printf(">>> Create in.csv\n");
// auto incsv = std::ofstream{"in.csv"};
// incsv << "#t_ns,in_ts" << std::endl;
//   int64_t t_ns = img->t_ns;
//   int64_t in_ns = std::chrono::steady_clock::now().time_since_epoch().count();
//   incsv << t_ns << "," << in_ns << std::endl;
// printf(">>> Close in.csv\n");
// incsv.close();

// printf(">>> Create out.csv\n");
// auto outcsv = std::ofstream{"out.csv"};
// outcsv << "#t_ns,out_ts" << std::endl;
//   int64_t t_ns = img->t_ns;
//   int64_t out_ns = std::chrono::steady_clock::now().time_since_epoch().count();
//   outcsv << t_ns << "," << out_ns << std::endl;
// printf(">>> Close out.csv\n");
// outcsv.close();export ds=/media/mateo/1A70DA1470D9F68B/monado-slam-datasets/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO02_hand_puncher_2

"""
