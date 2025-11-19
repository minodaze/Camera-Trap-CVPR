#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle exposure view for rare classes, using EXPOSURE RATIO (total_exposure / total_train_images).

For each dataset:

Top subplot:
    - rare ZS performance vs checkpoint (e.g., balanced_accuracy)
    - rare Best Oracle performance vs checkpoint
    - mean exposure ratio of rare classes (right y-axis, in %)
    - mean exposure ratio of test + rare classes (right y-axis, in %)

Middle subplot:
    - stacked bar chart of exposure ratio bins for RARE images at each checkpoint:
        * no_exposure : ratio == 0
        * weak        : (0, 1%]
        * moderate    : (1%, 10%]
        * strong      : (10%, 30%]
        * very_strong : > 30%

Bottom subplot:
    - stacked bar chart of exposure ratio bins for ALL eval images (test + rare) at each checkpoint
      using the same bins.

Here TOTAL exposure is defined for class c as:
    total_exposure(c) = sum_t train_count(c, t)
and exposure_ratio(c) = total_exposure(c) / total_train_images,
where total_train_images sums all train examples across all classes and checkpoints.
This matches Oracle training, which uses all ckps jointly.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import math
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

# For exposure (train/rare/test)
BASE_DIR = Path("/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare")
JSON_SUBDIR = "30"

# For rare performance logs
LOG_ROOT = Path("/fs/ess/PAS2099/camera-trap-CVPR-logs")

# Output directory for figures
FIG_OUT_DIR = Path("./oracle_exposure_figs")

# Datasets to process
DATASETS = [
    "serengeti/serengeti_H03",
    "serengeti/serengeti_E05",
    "APN/APN_13U",
    "serengeti/serengeti_L10",
    "serengeti/serengeti_D02",
    "serengeti/serengeti_T10",
    "serengeti/serengeti_Q07",
    "serengeti/serengeti_S11",
    "ENO/ENO_C04",
    "MTZ/MTZ_D03",
]

# Which metric from the rare logs to use: "balanced_accuracy" or "accuracy"
RARE_METRIC_KEY = "balanced_accuracy"

# Exposure RATIO bins (total_exposure / total_train_images)
#   bin 0: ratio == 0
#   bin 1: 0 < ratio <= 0.01   (<= 1%)
#   bin 2: 0.01 < ratio <= 0.10 (1%–10%)
#   bin 3: 0.10 < ratio <= 0.30 (10%–30%)
#   bin 4: ratio > 0.30         (> 30%)
EXPOSURE_BINS = [0.0, 0.01, 0.10, 0.30]
BIN_LABELS = ["no_exposure", "weak", "moderate", "strong", "very_strong"]


# ---------------- HELPERS ----------------

def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        print(f"[WARN] Missing file: {path}")
        return {}
    with path.open("r") as f:
        return json.load(f)


def ckp_name_to_index(ckp_name: str) -> int:
    """Convert 'ckp_3' -> 3 (large number fallback for weird keys)."""
    try:
        return int(ckp_name.split("_")[1])
    except Exception:
        return 10**9


def build_train_counts(train_data: Dict[str, List[Dict[str, Any]]]) -> Dict[int, Dict[int, int]]:
    """
    Build train_counts[class_id][t] = number of train images of class_id at checkpoint t.
    """
    train_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for ckp_name, entries in train_data.items():
        if not ckp_name.startswith("ckp_"):
            continue
        t = ckp_name_to_index(ckp_name)
        for e in entries:
            cid = int(e["class_id"])
            train_counts[cid][t] += 1

    return train_counts


def compute_total_exposure(
    cid: int,
    train_counts: Dict[int, Dict[int, int]],
) -> int:
    """
    Total exposure for class cid across ALL train checkpoints:
        total_exposure(c) = sum_t train_counts[c][t]
    """
    if cid not in train_counts:
        return 0
    return sum(train_counts[cid].values())


def compute_total_train_images(train_counts: Dict[int, Dict[int, int]]) -> int:
    """
    Total number of train images across all classes and checkpoints.
    Used to normalize total_exposure into an exposure ratio.
    """
    total = 0
    for _, per_ckp in train_counts.items():
        total += sum(per_ckp.values())
    return total


def assign_exposure_bin(val: float) -> int:
    """
    Assign exposure RATIO value to bin index 0..4 based on EXPOSURE_BINS.

    Bins:
        0: val == 0
        1: 0 < val <= 0.01
        2: 0.01 < val <= 0.10
        3: 0.10 < val <= 0.30
        4: val > 0.30
    """
    if val <= 0.0:
        return 0
    if val <= EXPOSURE_BINS[1]:
        return 1
    if val <= EXPOSURE_BINS[2]:
        return 2
    if val <= EXPOSURE_BINS[3]:
        return 3
    return 4


def load_rare_perf_curve(json_path: Path, metric_key: str) -> Dict[int, float]:
    """
    Load rare performance summary JSON and return:
        perf[T] = metric (e.g., balanced_accuracy) for checkpoint T
    """
    data = load_json(json_path)
    if not data:
        return {}

    curve: Dict[int, float] = {}

    if "checkpoint_results" in data:
        items = data["checkpoint_results"].items()
    else:
        items = [(k, v) for k, v in data.items() if k.startswith("ckp_")]

    for ckp_name, stats in items:
        if not ckp_name.startswith("ckp_"):
            continue
        T = ckp_name_to_index(ckp_name)
        if metric_key in stats:
            curve[T] = float(stats[metric_key])

    return curve


# ---------------- MAIN PER-DATASET ANALYSIS ----------------

def analyze_and_plot_dataset(dataset: str) -> None:
    # ---------- Paths for exposure ----------
    json_dir = BASE_DIR / dataset / JSON_SUBDIR
    train_path = json_dir / "train.json"
    test_path = json_dir / "test.json"
    rare_path = json_dir / "rare.json"

    print(f"\n=== Dataset (Oracle view, ratio): {dataset} ===")
    print(f"Using train: {train_path}")
    print(f"Using test : {test_path}")
    print(f"Using rare : {rare_path}")

    train_data = load_json(train_path)
    test_data = load_json(test_path)
    rare_data = load_json(rare_path)

    if not train_data or not test_data or not rare_data:
        print(f"[WARN] Missing train, test, or rare for dataset {dataset}. Skipping.")
        return

    # Build train counts: class_id -> {t -> count}
    train_counts = build_train_counts(train_data)
    total_train_images = compute_total_train_images(train_counts)
    if total_train_images == 0:
        print(f"[WARN] total_train_images == 0 for {dataset}. Skipping.")
        return

    # Collect all checkpoints that appear in rare_data (we plot at these)
    ckps = sorted(
        [ckp for ckp in rare_data.keys() if ckp.startswith("ckp_")],
        key=ckp_name_to_index,
    )
    if not ckps:
        print(f"[WARN] No checkpoints with rare data for dataset {dataset}. Skipping.")
        return

    # ---------- Load rare performance curves ----------
    tag = dataset.replace("/", "_")

    zs_log = (
        LOG_ROOT
        / "rare_zs"
        / tag
        / "bioclip2"
        / "full_text_head"
        / "log"
        / "final_training_summary.json"
    )
    oracle_log = (
        LOG_ROOT
        / "rare_best_oracle"
        / tag
        / "bioclip2"
        / "lora_8_text_head"
        / "log"
        / "final_training_summary.json"
    )
    accum_log = (
        LOG_ROOT
        / "rare_best_accum"
        / tag
        / "bioclip2"
        / "lora_8_text_head"
        / "all"
        / "log"
        / "final_training_summary.json"
    )

    print(f"  Rare ZS log     : {zs_log}")
    print(f"  Rare Oracle log : {oracle_log}")
    print(f"  Rare Accum log  : {accum_log}")

    zs_curve = load_rare_perf_curve(zs_log, RARE_METRIC_KEY)
    oracle_curve = load_rare_perf_curve(oracle_log, RARE_METRIC_KEY)
    accum_curve = load_rare_perf_curve(accum_log, RARE_METRIC_KEY)

    # ---------- Compute exposure ratio stats per checkpoint ----------
    mean_ratio_exposure_rare: Dict[int, float] = {}
    mean_ratio_exposure_all: Dict[int, float] = {}  # NEW: mean exposure ratio for test+rare

    bin_counts_rare: Dict[int, List[int]] = {}
    bin_counts_all: Dict[int, List[int]] = {}
    total_rare_per_ckp: Dict[int, int] = {}
    total_eval_per_ckp: Dict[int, int] = {}

    for ckp_name in ckps:
        T = ckp_name_to_index(ckp_name)
        rare_entries = rare_data.get(ckp_name, [])
        test_entries = test_data.get(ckp_name, [])

        # ----- Rare-only stats -----
        ratio_values_rare = []
        bins_rare = [0] * len(BIN_LABELS)

        for e in rare_entries:
            cid = int(e["class_id"])
            total_exp = compute_total_exposure(cid, train_counts)
            ratio = total_exp / total_train_images if total_train_images > 0 else 0.0
            ratio_values_rare.append(ratio)

            bidx = assign_exposure_bin(ratio)
            bins_rare[bidx] += 1

        if ratio_values_rare:
            mean_ratio_rare = sum(ratio_values_rare) / len(ratio_values_rare)
        else:
            mean_ratio_rare = 0.0

        mean_ratio_exposure_rare[T] = mean_ratio_rare
        bin_counts_rare[T] = bins_rare
        total_rare_per_ckp[T] = len(rare_entries)

        # ----- Combined test + rare stats -----
        all_entries = list(test_entries) + list(rare_entries)
        bins_all = [0] * len(BIN_LABELS)
        ratio_values_all = []  # NEW: store ratios for all eval images at T

        for e in all_entries:
            cid = int(e["class_id"])
            total_exp = compute_total_exposure(cid, train_counts)
            ratio = total_exp / total_train_images if total_train_images > 0 else 0.0
            ratio_values_all.append(ratio)

            bidx = assign_exposure_bin(ratio)
            bins_all[bidx] += 1

        if ratio_values_all:
            mean_ratio_all = sum(ratio_values_all) / len(ratio_values_all)
        else:
            mean_ratio_all = 0.0

        mean_ratio_exposure_all[T] = mean_ratio_all   # NEW
        bin_counts_all[T] = bins_all
        total_eval_per_ckp[T] = len(all_entries)

    # ---------- Prepare x-axis and series ----------
    Ts = sorted(mean_ratio_exposure_rare.keys())
    mean_ratio_rare_list = [mean_ratio_exposure_rare[T] for T in Ts]
    mean_ratio_all_list = [mean_ratio_exposure_all[T] for T in Ts]

    mean_ratio_rare_pct_list = [r * 100.0 for r in mean_ratio_rare_list]
    mean_ratio_all_pct_list = [r * 100.0 for r in mean_ratio_all_list]

    # Rare performance: only plot where we have data
    zs_T = [T for T in Ts if T in zs_curve]
    zs_y = [zs_curve[T] for T in zs_T]
    accum_T = [T for T in Ts if T in accum_curve]
    accum_y = [accum_curve[T] for T in accum_T]

    oracle_T = [T for T in Ts if T in oracle_curve]
    oracle_y = [oracle_curve[T] for T in oracle_T]

    # Convert bin_counts to fractions per checkpoint (rare-only)
    bin_fracs_rare = {b: [] for b in range(len(BIN_LABELS))}
    for T in Ts:
        total = total_rare_per_ckp[T]
        counts = bin_counts_rare[T]
        if total > 0:
            fracs = [c / total for c in counts]
        else:
            fracs = [0.0] * len(BIN_LABELS)
        for b in range(len(BIN_LABELS)):
            bin_fracs_rare[b].append(fracs[b])

    # Convert bin_counts_all to fractions per checkpoint (test + rare)
    bin_fracs_all = {b: [] for b in range(len(BIN_LABELS))}
    for T in Ts:
        total = total_eval_per_ckp[T]
        counts = bin_counts_all[T]
        if total > 0:
            fracs = [c / total for c in counts]
        else:
            fracs = [0.0] * len(BIN_LABELS)
        for b in range(len(BIN_LABELS)):
            bin_fracs_all[b].append(fracs[b])

    # ---------------- PLOT ----------------
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_OUT_DIR / f"oracle_exposure_ratio_{tag}.png"

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.6, 1.6]},
    )

    # ----- Top: rare ZS & Oracle (left-y) + mean exposure ratios (right-y) -----
    ax_perf = ax1
    ax_exp = ax1.twinx()

    ax_perf.plot(zs_T, zs_y, marker="o", linestyle="-", label="rare ZS")
    ax_perf.plot(oracle_T, oracle_y, marker="s", linestyle="-", label="rare Best Oracle")
    ax_perf.plot(accum_T, accum_y, marker="^", linestyle="-", label="rare Best Accum")

    ax_perf.set_ylabel(f"Rare {RARE_METRIC_KEY}", fontsize=11)
    ax_perf.set_ylim(0.0, 1.0)
    ax_perf.grid(True, axis="y", linestyle="--", alpha=0.4)

    RARE_COLOR = "#D62728"      # red (matplotlib default red)
    RARETEST_COLOR = "#9467BD"  # purple (matplotlib default purple)

    # Right y-axis: two lines – rare-only and test+rare
    ax_exp.plot(
        Ts,
        mean_ratio_rare_pct_list,
        marker="d",
        linestyle="--",
        label="Mean exposure ratio (rare)",
        alpha=0.7,
        color=RARE_COLOR,
    )
    ax_exp.plot(
        Ts,
        mean_ratio_all_pct_list,
        marker="^",
        linestyle="--",
        label="Mean exposure ratio (test + rare)",
        alpha=0.7,
        color=RARETEST_COLOR,
    )
    ax_exp.set_ylabel("Mean exposure ratio (%)", fontsize=11)
    ax_exp.set_ylim(0, 50)

    ax_perf.set_title(f"Exposure Ratio & Rare Performance (Oracle) for {dataset}", fontsize=13)

    lines_perf, labels_perf = ax_perf.get_legend_handles_labels()
    lines_exp, labels_exp = ax_exp.get_legend_handles_labels()
    ax_perf.legend(
        lines_perf + lines_exp,
        labels_perf + labels_exp,
        loc="upper right",
        fontsize=9,
    )

    # ----- Middle: stacked bar of exposure ratio bins (RARE only) -----
    bottoms = [0.0] * len(Ts)
    for bidx, label in enumerate(BIN_LABELS):
        vals = bin_fracs_rare[bidx]
        ax2.bar(Ts, vals, bottom=bottoms, label=label)
        bottoms = [bottoms[i] + vals[i] for i in range(len(Ts))]

    ax2.set_ylabel("Fraction of rare images", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(
        title="Exposure ratio bin (rare)\n(total_exposure / total_train_images)",
        fontsize=9,
        title_fontsize=10,
        ncol=3,
        loc="upper right",
    )

    # ----- Bottom: stacked bar of exposure ratio bins (TEST + RARE) -----
    bottoms = [0.0] * len(Ts)
    for bidx, label in enumerate(BIN_LABELS):
        vals = bin_fracs_all[bidx]
        ax3.bar(Ts, vals, bottom=bottoms, label=label)
        bottoms = [bottoms[i] + vals[i] for i in range(len(Ts))]

    ax3.set_xlabel("Checkpoint index (T)", fontsize=11)
    ax3.set_ylabel("Fraction of test + rare images", fontsize=11)
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax3.legend(
        title="Exposure ratio bin (test + rare)\n(total_exposure / total_train_images)",
        fontsize=9,
        title_fontsize=10,
        ncol=3,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved figure: {fig_path}")


def main():
    for ds in DATASETS:
        analyze_and_plot_dataset(ds)


if __name__ == "__main__":
    main()
