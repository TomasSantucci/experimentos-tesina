#!/usr/bin/env python
# pylint: disable-all

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.pyplot as plt

import numpy as np
import json

CAUSAL = True # Set to False to show fully postprocessed bundle adjustment results]
BATCH_DIR = "../.."

UNK = -1234567890  # (?) Represents an unknown value, latest data doesnt have it
NA = -1234567891  # (—)

DATASETS = ["MIO01","MIO02","MIO03","MIO04","MIO05","MIO06","MIO07","MIO08","MIO09","MIO10","MIO11","MIO12","MIO13","MIO14","MIO15","MIO16","MIPB01","MIPB02","MIPB03","MIPB04","MIPB05","MIPB06","MIPB07","MIPB08","MIPP01","MIPP02","MIPP03","MIPP04","MIPP05","MIPP06","MIPT01","MIPT02","MIPT03","MGO01","MGO02","MGO03","MGO04","MGO05","MGO06","MGO07","MGO08","MGO09","MGO10","MGO11","MGO12","MGO13","MGO14","MGO15","MOO01","MOO02","MOO03","MOO04","MOO05","MOO06","MOO07","MOO08","MOO09","MOO10","MOO11","MOO12","MOO13","MOO14","MOO15","MOO16"]  # fmt: skip
assert len(set(DATASETS)) == 64


def load_storage(systems):
    # Load all system numbers into a dict storage
    storage = {}
    metrics = ["ate", "rte", "success", "completion"]
    load_files = [f"{BATCH_DIR}/{metric}.{system}.json" for metric in metrics for system in systems]
    for load_file in load_files:
        with open(load_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for metric, runs in loaded.items():  # Merge loaded files
                if metric not in storage:
                    storage[metric] = {}
                for run_name, run in runs.items():
                    for ds in DATASETS:
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


def select_index(ds, completion, success, ate_avg, ate_std, rte_avg, rte_std):
    # Select by most successful frames
    candidates = list(range(len(success)))
    max_success = max(success)

    if success.count(max_success) == 1:  # Found it
        return success.index(max_success)
    elif max_success in [NA, UNK]:  # Any will do
        return 0

    # Success tie, pick lowest ATE
    candidates = [i for i, s in enumerate(success) if s == max_success]
    ate_avg = [a if a not in [NA, UNK] else np.inf for a in ate_avg]  # Convert bad values to inf
    min_err = min([ate_avg[i] for i in candidates])

    if list(ate_avg).count(min_err) == 1:
        return ate_avg.index(min_err)

    # ATE tie, pick lowest RTE
    candidates = [i for i, a in enumerate(ate_avg) if a == min_err and success[i] == max_success]
    rte_avg = [r if r not in [NA, UNK] else np.inf for r in rte_avg]  # Convert bad values to inf
    min_err = min([rte_avg[i] for i in candidates])

    if list(rte_avg).count(min_err) == 1:
        return rte_avg.index(min_err)

    raise NotImplementedError(f"Could not select index for {ds}, a further tie breaker is needed")


def get_basalt_results():
    SYSTEMS = ["basalt"]
    DETERMINISTIC_RUN = "basalt.det"

    def select_run_basalt(storage, ds):
        assert ds in storage["completion"][DETERMINISTIC_RUN]
        return DETERMINISTIC_RUN  # Use deterministic run always

    storage = load_storage(SYSTEMS)
    selected_runs = {ds: select_run_basalt(storage, ds) for ds in DATASETS}
    return select_runs(storage, selected_runs)


def get_okvis2_results(causal=True):
    SYSTEMS = ["okvis2.vio" if causal else "okvis2.slamfinal"]
    RUNS = [f"{SYSTEMS[0]}.{i}" for i in [1, 2, 3]]
    storage = load_storage(SYSTEMS)
    selected_runs = {ds: select_run(storage, ds, RUNS) for ds in DATASETS}
    return select_runs(storage, selected_runs)


def get_orbslam3_results(causal=True):
    SYSTEMS = ["orbslam3.rt" if causal else "orbslam3.ba"]
    RUNS = [f"{SYSTEMS[0]}.{i}" for i in [1, 2, 3]]
    storage = load_storage(SYSTEMS)
    selected_runs = {ds: select_run(storage, ds, RUNS) for ds in DATASETS}
    return select_runs(storage, selected_runs)


def get_dmvio_results():
    SYSTEMS = ["dmvio.scaled"]
    RUNS = [f"{SYSTEMS[0]}.{i}" for i in [1, 2, 3]]
    storage = load_storage(SYSTEMS)
    selected_runs = {ds: select_run(storage, ds, RUNS) for ds in DATASETS}
    return select_runs(storage, selected_runs)


def get_snakeslam_results(causal=True):
    if not causal:
        SYSTEMS = ["snakeslam.ba"]
        RUNS = [f"{SYSTEMS[0]}.{i}" for i in ["det", "nd1", "nd2"]]
        storage = load_storage(SYSTEMS)
        selected_runs = {ds: select_run(storage, ds, RUNS) for ds in DATASETS}
        return select_runs(storage, selected_runs)

    SYSTEMS = ["snakeslam.causal", "snakeslam.rtvalid"]
    CAUSAL_RUN = "snakeslam.causal.det"
    RTVALID_RUN = "snakeslam.rtvalid.det"

    def select_run_snakeslam(storage, ds):
        # Prioritize "causal" over "rtvalid" run
        causal = storage["completion"][CAUSAL_RUN][ds]
        if causal not in [NA, UNK]:
            return CAUSAL_RUN

        rtvalid = storage["completion"][RTVALID_RUN][ds]
        if rtvalid not in [NA, UNK]:
            return RTVALID_RUN

        return CAUSAL_RUN

    storage = load_storage(SYSTEMS)
    selected_runs = {ds: select_run_snakeslam(storage, ds) for ds in DATASETS}
    return select_runs(storage, selected_runs)


basalt_results = get_basalt_results()
dmvio_results = get_dmvio_results()
orbslam3_results = get_orbslam3_results(CAUSAL)
okvis2_results = get_okvis2_results(CAUSAL)
snakeslam_results = get_snakeslam_results(CAUSAL)

# Set font to CMU Serif
plt.rcParams["font.family"] = ["CMU serif", "Sans-serif"]
plt.rcParams["pdf.fonttype"] = 42  # see: http://phyletica.org/matplotlib-fonts/
plt.rcParams["ps.fonttype"] = 42

# plt.rcParams["font.family"] = "CMU sans serif"
# plt.rcParams["font.family"] = "CMU mono"
# plt.rcParams["font.family"] = "Fira Code"
# plt.rcParams["font.family"] = "Roboto"
# plt.rcParams["font.family"] = "Josefin Sans"
# plt.rcParams["font.family"] = "Montserrat"


def create_cmap(colors=["#263238", "#ECEFF1"]):
    cmap = LinearSegmentedColormap.from_list("custom_rocket", colors)
    return cmap


# ate_cmap = create_cmap(["#0D47A1", "#E3F2FD"])  # blue
ate_cmap = create_cmap(["#274d87", "#E3F2FD"])  # desaturated blue

# rte_cmap = create_cmap(["#1B5E20", "#E8F5E9"])  # green
rte_cmap = create_cmap(["#2d4c30", "#E8F5E9"])  # desaturated green

# completions_cmap = create_cmap(["#ECEFF1", "#263238"])  # dark grey
completions_cmap = create_cmap(["#ECEFF1", "#4e5e6f"])  # lighter dark grey

successes_cmap = create_cmap(["#ECEFF1", "#4e5e6f"])  # lighter dark grey

# ate_cmap = create_cmap(["#B71C1C", "#FFEBEE"]) # red
# rte_cmap = create_cmap(["#006064", "#E0F7FA"]) # cyan
# completions_cmap = create_cmap(["#E3F2FD", "#0D47A1"]) # blue
# completions_cmap = create_cmap(["#E8F5E9", "#1B5E20"]) # green
# completions_cmap = create_cmap(["#ECEFF1", "#263238"])  # dark grey
# completions_cmap = create_cmap(["#EDE7F6", "#311B92"])  # Deep Purple
# completions_cmap = create_cmap(["#FFF8E1", "#FF6F00"])  # Amber
# completions_cmap = create_cmap(["#fff2c9", "#ffd966"])  # Amber

if CAUSAL:
    SYSTEMS = ["Basalt", "OKVIS2", "ORB-SLAM3", "DM-VIO", "SnakeSLAM"]
    RESULTS = [basalt_results, okvis2_results, orbslam3_results, dmvio_results, snakeslam_results]
else:
    SYSTEMS = ["OKVIS2", "ORB-SLAM3", "SnakeSLAM"]
    RESULTS = [okvis2_results, orbslam3_results, snakeslam_results]

assert len(DATASETS) == 64

# Sample data as before
ates = np.ones((len(SYSTEMS), len(DATASETS))) * -1
rtes = np.ones((len(SYSTEMS), len(DATASETS))) * -1
completions = np.ones((len(SYSTEMS), len(DATASETS))) * -1
successes = np.ones((len(SYSTEMS), len(DATASETS))) * -1

for ds in DATASETS:
    for system in SYSTEMS:
        idx = SYSTEMS.index(system)
        ates[idx, DATASETS.index(ds)] = RESULTS[idx]["ates_avg"][ds]
        rtes[idx, DATASETS.index(ds)] = RESULTS[idx]["rtes_avg"][ds]
        completions[idx, DATASETS.index(ds)] = RESULTS[idx]["completion"][ds]
        successes[idx, DATASETS.index(ds)] = RESULTS[idx]["success"][ds]

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
M_SCALER = 100  # 100 for cm, 1 for m
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

# Print median values
for system, ate, rte, completion, success in zip(SYSTEMS, ate_median, rte_median, completion_median, success_median):
    print(f"{system=}, {ate=:.2f}, {rte=:.2f}, {completion=:.0f}, {success=:.0f}")

# Get values below threshold
ATE_THRESHOLD = 10  # cm
RTE_THRESHOLD = 1  # cm
COMPLETION_THRESHOLD = 98
SUCCESS_THRESHOLD = 98
n = len(DATASETS)
ate_ltth = np.array([sum(sys_ates < ATE_THRESHOLD) / n for sys_ates in ates])
rte_ltth = np.array([sum(sys_rtes < RTE_THRESHOLD) / n for sys_rtes in rtes])
comp_ltth = np.array([sum(sys_comp > COMPLETION_THRESHOLD) / n for sys_comp in completions])
succ_ltth = np.array([sum(sys_succ > SUCCESS_THRESHOLD) / n for sys_succ in successes])

print(f"{ate_ltth=}")
print(f"{rte_ltth=}")
print(f"{comp_ltth=}")
print(f"{succ_ltth=}")

ATE_DIVERGE_FROM = 10 * M_SCALER  # 10m
ates_sorted = sorted(ates.reshape((-1,)))
ates_sorted = [x for x in ates_sorted if x <= ATE_DIVERGE_FROM]
ate_p90 = ates_sorted[int(len(ates_sorted) * 0.9)]

RTE_DIVERGE_FROM = 0.1 * M_SCALER  # 10cm
rtes_sorted = sorted(rtes.reshape((-1,)))
rtes_sorted = [x for x in rtes_sorted if x <= RTE_DIVERGE_FROM]
rte_p90 = rtes_sorted[int(len(rtes_sorted) * 0.9)]

SUCCESS_FROM = 98  # From 98% we just print the checkmark


U = abs(UNK)
N = abs(NA)
ates = np.nan_to_num(ates, nan=U, posinf=N)
rtes = np.nan_to_num(rtes, nan=U, posinf=N)
completions = np.nan_to_num(completions, nan=-U, neginf=-N)
successes = np.nan_to_num(successes, nan=-U, neginf=-N)

# ate_min = 0
# rte_min = 0

ate_min = min(ates_sorted)
rte_min = min(rtes_sorted)

# ate_max = 1.5 * M_SCALER
# rte_max = 0.05 * M_SCALER

ate_max = ate_p90
rte_max = rte_p90
print(f"{ate_max=}")
print(f"{rte_max=}")

# Add median row
ates = np.hstack([ates, ate_median.reshape(len(SYSTEMS), 1)])
rtes = np.hstack([rtes, rte_median.reshape(len(SYSTEMS), 1)])
completions = np.hstack([completions, completion_median.reshape(len(SYSTEMS), 1)])
successes = np.hstack([successes, success_median.reshape(len(SYSTEMS), 1)])
DATASETS += ["Median"]

# Add lower-than-threshold row
# ates = np.hstack([ates, ate_ltth.reshape(len(SYSTEMS), 1)])
# rtes = np.hstack([rtes, rte_ltth.reshape(len(SYSTEMS), 1)])
# completions = np.hstack([completions, comp_ltth.reshape(len(SYSTEMS), 1)])
# successes = np.hstack([successes, succ_ltth.reshape(len(SYSTEMS), 1)])
# DATASETS += ["Accept"]


def plot_heatmaps(ates, rtes, completions, successes, dataset_names, system_names):
    # fig, axes = plt.subplots(1, 3, figsize=(12, 18), constrained_layout=True)
    fig, axes = plt.subplots(1, 3, figsize=(8, 24), constrained_layout=True)
    fig.set_size_inches(6.4638, 13.625625)  # Arbitrary size that doesnt look to tight

    # cbar_kws = {'shrink': 0.2, 'label': '', 'ticks': []}  # Customize as needed
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
        # cmap="cool",
        # cmap="rocket_r",
        cmap=ate_cmap,
        annot=annotations_ate,
        # annot=True,
        # fmt=".2f",
        fmt="",
        vmin=ate_min,
        vmax=ate_max,
        xticklabels=system_names,
        yticklabels=dataset_names,
        cbar_kws=cbar_kws | {"ticks": [ate_min, ate_max]},
    )
    # axes[0].tick_params(axis='x', labelsize=8, labelbottom=False, bottom=False, top=False, labeltop=True)
    axes[0].tick_params(**tick_params_kws)
    # axes[0].set_xticks([ate_min, ate_max])

    # axes[0].set_title("Absolute Trajectory Error [cm]\n", fontsize=10)
    axes[0].set_title("ATE [cm] (SE3 aligned)", fontsize=10)
    # axes[0].set_xticks(
    #     np.arange(0, len(dataset_names), step=5)
    # )  # Label every 5 datasets for clarity

    # Plotting Relative Trajectory Error (RTE) heatmap

    annotate_rte = lambda x: ("×" if x == U else "×" if x == N else "∞" if x > RTE_DIVERGE_FROM else f"{x:.2f}")
    annotations_rte = np.vectorize(annotate_rte)(rtes.T)
    sns.heatmap(
        rtes.T,
        ax=axes[1],
        # cmap="YlOrRd",
        # cmap="mako_r",
        # cmap=custom_rocket_colormap(final_dark_color="#261226"),
        cmap=rte_cmap,
        annot=annotations_rte,
        # annot=True,
        # fmt=".3f",
        fmt="",
        vmin=rte_min,
        vmax=rte_max,
        xticklabels=system_names,
        yticklabels=[],
        cbar_kws=cbar_kws | {"ticks": [rte_min, rte_max]},
    )
    axes[1].tick_params(**tick_params_kws)
    # axes[1].set_title("Relative Trajectory Error [cm]\n", fontsize=10)
    axes[1].set_title("RTE [cm] (Δ = 6 frames)", fontsize=10)

    # Plotting Success Percentage heatmap
    annotate_succ = lambda x: ("×" if x == -U else "×" if x == -N else "✓" if x > SUCCESS_FROM else f"{x:.0f}")
    annotations_succ = np.vectorize(annotate_succ)(successes.T)
    sns.heatmap(
        successes.T,
        ax=axes[2],
        # cmap="winter",
        # cmap="crest_r",
        cmap=successes_cmap,
        # annot=True,
        annot=annotations_succ,
        # fmt=".0f",
        fmt="",
        vmin=0,
        vmax=100,
        xticklabels=system_names,
        yticklabels=[],
        cbar_kws=cbar_kws | {"ticks": [0, 100]},
    )
    axes[2].tick_params(**tick_params_kws)
    axes[2].set_title("Completed frames [%]", fontsize=10)
    # axes[2].set_xticks(
    #     np.arange(0, len(dataset_names), step=5)
    # )  # Label every 5 datasets for clarity

    plt.show()


def main():
    plot_heatmaps(ates, rtes, completions, successes, DATASETS, SYSTEMS)
    print(
        """
        Now edit in inkscape and remember to do the following:
        1. Add ¹,²,³ footnotes
        2. Redistribute titles to have more proportional distribution
        3. Split the 4 subcategories
        4. Move columns horizontally closer together
        5. Add median row
        6. Move and edit cmap labels around a bit
        7. Get cmap closer
        """
    )


if __name__ == "__main__":
    main()
