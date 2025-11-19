#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For each dataset, compute exposure for rare classes and
plot them together with rare zs / rare best_oracle / rare best_accum performance.

Definitions
-----------
For a rare image with class c at checkpoint T:

Past exposure (for Accum):
    past_exposure(c, T) = sum_{t < T} train_count(c, t)

We then aggregate over all rare images at each checkpoint T:

Top subplot:
    - Left y-axis: rare balanced accuracy vs checkpoint
        * rare_zs
        * rare_best_oracle
        * rare_best_accum
    - Right y-axis: mean past_exposure for rare classes at each checkpoint.

Bottom subplot:
    - Stacked bar plot of fraction of rare images in past_exposure bins:
        [0],
        (0, 1],
        (1, 5],
        (5, 20],
        (20, +inf)

Inputs
------
Exposure:
    {BASE_DIR}/{dataset}/{JSON_SUBDIR}/train.json
    {BASE_DIR}/{dataset}/{JSON_SUBDIR}/rare.json

Rare performance logs:
    {LOG_ROOT}/rare_zs/{TAG}/bioclip2/full_text_head/log/final_training_summary.json
    {LOG_ROOT}/rare_best_oracle/{TAG}/bioclip2/lora_8_text_head/log/final_training_summary.json
    {LOG_ROOT}/rare_best_accum/{TAG}/bioclip2/lora_8_text_head/all/log/final_training_summary.json

Where:
    dataset (e.g.): "APN/APN_13U"
    TAG = dataset.replace("/", "_")  (e.g. "APN_APN_13U")

Outputs
-------
One PNG per dataset (under FIG_OUT_DIR) named:
    exposure_quality_{TAG}.png
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

# For exposure (train/rare)
BASE_DIR = Path("/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare")
JSON_SUBDIR = "30"

# For rare performance logs
LOG_ROOT = Path("/fs/ess/PAS2099/camera-trap-CVPR-logs")

# Output directory for figures
FIG_OUT_DIR = Path("./accum_exposure_figs")

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

# Past exposure bins (on raw counts, no temporal decay)
#   bin 0: [0]
#   bin 1: (0, 1]
#   bin 2: (1, 5]
#   bin 3: (5, 20]
#   bin 4: (20, +inf)
# EXPOSURE_BINS = [0.0, 1.0, 5.0, 20.0]
EXPOSURE_BINS = [0.0, 5.0, 20.0, 50.0]
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


def compute_past_exposure(
    cid: int,
    T: int,
    train_counts: Dict[int, Dict[int, int]],
) -> int:
    """
    Compute PAST exposure for class cid at checkpoint T.

    past_exposure = sum_{t < T} train_count(c, t)

    This matches your best_accum(T) training, which uses all ckps < T together.
    """
    if cid not in train_counts:
        return 0

    raw = 0
    for t_train, count in train_counts[cid].items():
        if t_train < T:
            raw += count
    return raw


def assign_exposure_bin(val: float) -> int:
    """
    Assign past_exposure value to bin index 0..4 based on EXPOSURE_BINS.

    Bins:
        0: val == 0
        1: 0 < val <= 1
        2: 1 < val <= 5
        3: 5 < val <= 20
        4: val > 20
    """
    if val <= 0:
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

    Supports two formats:

    1) {
           "checkpoint_results": {
               "ckp_1": {...},
               "ckp_2": {...},
               ...
           },
           "averages": {...}
       }

    2) {
           "ckp_1": {...},
           "ckp_2": {...},
           ...
           "averages": {...}
       }
    """
    data = load_json(json_path)
    if not data:
        return {}

    curve: Dict[int, float] = {}

    if "checkpoint_results" in data:
        # Format 1
        items = data["checkpoint_results"].items()
    else:
        # Format 2: ckps at top level, skip 'averages' and any non-ckp keys
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
    rare_path = json_dir / "rare.json"

    print(f"\n=== Dataset: {dataset} ===")
    print(f"Using train: {train_path}")
    print(f"Using rare : {rare_path}")

    train_data = load_json(train_path)
    rare_data = load_json(rare_path)

    if not train_data or not rare_data:
        print(f"[WARN] Missing train or rare for dataset {dataset}. Skipping.")
        return

    # Build train counts: class_id -> {t -> count}
    train_counts = build_train_counts(train_data)

    # Collect all ckps that appear in rare_data (since exposure is only meaningful there)
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

    print(f"  Rare ZS log      : {zs_log}")
    print(f"  Rare Oracle log  : {oracle_log}")
    print(f"  Rare Accum log   : {accum_log}")

    zs_curve = load_rare_perf_curve(zs_log, RARE_METRIC_KEY)
    oracle_curve = load_rare_perf_curve(oracle_log, RARE_METRIC_KEY)
    accum_curve = load_rare_perf_curve(accum_log, RARE_METRIC_KEY)

    # For each checkpoint, compute:
    #   - mean_past_exposure[T]
    #   - bin_counts[T][bin_idx]
    mean_past_exposure: Dict[int, float] = {}
    bin_counts: Dict[int, List[int]] = {}
    total_rare_per_ckp: Dict[int, int] = {}

    for ckp_name in ckps:
        T = ckp_name_to_index(ckp_name)
        rare_entries = rare_data.get(ckp_name, [])

        exposure_values = []
        bin_count = [0] * len(BIN_LABELS)

        for e in rare_entries:
            cid = int(e["class_id"])
            past_exp = compute_past_exposure(cid, T, train_counts)
            exposure_values.append(past_exp)

            bidx = assign_exposure_bin(past_exp)
            bin_count[bidx] += 1

        mean_past = sum(exposure_values) / len(exposure_values) if exposure_values else 0.0

        mean_past_exposure[T] = mean_past
        bin_counts[T] = bin_count
        total_rare_per_ckp[T] = len(rare_entries)

    # Prepare x-axis checkpoints (T values)
    Ts = sorted(mean_past_exposure.keys())

    # Build lists for plotting
    mean_past_list = [mean_past_exposure[T] for T in Ts]

    # Rare performance (aligned with Ts; use NaN if missing)
    zs_vals = [zs_curve.get(T, math.nan) for T in Ts]
    oracle_vals = [oracle_curve.get(T, math.nan) for T in Ts]
    accum_vals = [accum_curve.get(T, math.nan) for T in Ts]

    # Convert bin_counts to fractions per checkpoint
    bin_fracs = {b: [] for b in range(len(BIN_LABELS))}  # bin_idx -> list over Ts

    for T in Ts:
        total = total_rare_per_ckp[T]
        counts = bin_counts[T]
        if total > 0:
            fracs = [c / total for c in counts]
        else:
            fracs = [0.0] * len(BIN_LABELS)
        for b in range(len(BIN_LABELS)):
            bin_fracs[b].append(fracs[b])

    # ---------------- PLOT ----------------
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_OUT_DIR / f"exposure_quality_{tag}.png"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.6]}
    )

    # ----- Top: rare performance (left-y) + mean past exposure (right-y) -----
    ax_perf = ax1
    ax_exp = ax1.twinx()

    # Rare performance (balanced accuracy or accuracy)
    ax_perf.plot(Ts, zs_vals, marker="o", linestyle="-", label="rare ZS")
    ax_perf.plot(Ts, oracle_vals, marker="s", linestyle="-", label="rare Best Oracle")
    ax_perf.plot(Ts, accum_vals, marker="^", linestyle="-", label="rare Best Accum")

    ax_perf.set_ylabel(f"Rare {RARE_METRIC_KEY}", fontsize=11)
    ax_perf.set_ylim(0.0, 1.0)
    ax_perf.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Mean past exposure on secondary axis
    ax_exp.plot(
        Ts,
        mean_past_list,
        marker="d",
        linestyle="--",
        label="Mean past exposure (ckps < T)",
        alpha=0.7,
    )
    ax_exp.set_ylabel("Mean past exposure\n(# train images before T)", fontsize=11)

    # Title & combined legend
    ax_perf.set_title(f"Past Exposure & Rare Performance for {dataset}", fontsize=13)

    # Build a combined legend
    lines_perf, labels_perf = ax_perf.get_legend_handles_labels()
    lines_exp, labels_exp = ax_exp.get_legend_handles_labels()
    ax_perf.legend(
        lines_perf + lines_exp,
        labels_perf + labels_exp,
        loc="upper right",
        fontsize=9,
    )

    # ----- Bottom: stacked bar of past exposure bins -----
    bottoms = [0.0] * len(Ts)
    for bidx, label in enumerate(BIN_LABELS):
        vals = bin_fracs[bidx]
        ax2.bar(Ts, vals, bottom=bottoms, label=label)
        bottoms = [bottoms[i] + vals[i] for i in range(len(Ts))]

    ax2.set_xlabel("Checkpoint index (T)", fontsize=11)
    ax2.set_ylabel("Fraction of rare images", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(
        title="Past exposure bin\n(sum train counts over ckps < T)",
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
