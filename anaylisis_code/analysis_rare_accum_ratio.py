#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For each dataset, compute exposure *ratio* for rare classes and
plot them together with rare zs / rare best_oracle / rare best_accum performance.

Definitions
-----------
For an eval image (rare or test) with class c at checkpoint T:

Past exposure (for Accum, in counts):
    past_exposure(c, T) = sum_{t < T} train_count(c, t)

Total train seen before T (all classes):
    total_train_before_T(T) = sum_{c'} sum_{t < T} train_count(c', t)

Past exposure ratio:
    past_ratio(c, T) = past_exposure(c, T) / total_train_before_T(T)

We then aggregate over rare-only and (test+rare) at each checkpoint T:

Top subplot:
    - Left y-axis: rare balanced accuracy vs checkpoint
        * rare_zs
        * rare_best_oracle
        * rare_best_accum
    - Right y-axis:
        * mean past_ratio for rare images at each checkpoint (in %)
        * mean past_ratio for (test + rare) images at each checkpoint (in %)

Middle subplot:
    - Stacked bar plot of fraction of RARE images in past_ratio bins.

Bottom subplot:
    - Stacked bar plot of fraction of (TEST + RARE) images in past_ratio bins.

Bins (on past_ratio):
    [0],
    (0, 0.01]   (~0–1%)
    (0.01, 0.10] (~1–10%)
    (0.10, 0.30] (~10–30%)
    (0.30, +inf)

Inputs
------
Exposure:
    {BASE_DIR}/{dataset}/{JSON_SUBDIR}/train.json
    {BASE_DIR}/{dataset}/{JSON_SUBDIR}/test.json
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
    exposure_quality_ratio_{TAG}.png
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

# For exposure (train/test/rare)
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

# Past exposure *ratio* bins (past_ratio = past_exposure / total_train_before_T)
#   bin 0: ratio == 0
#   bin 1: 0     < ratio <= 0.01   (<= 1%)
#   bin 2: 0.01  < ratio <= 0.10   (1%–10%)
#   bin 3: 0.10  < ratio <= 0.30   (10%–30%)
#   bin 4: ratio > 0.30            (> 30%)
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


def compute_past_exposure(
    cid: int,
    T: int,
    train_counts: Dict[int, Dict[int, int]],
) -> int:
    """
    Compute PAST exposure (in counts) for class cid at checkpoint T.

    past_exposure = sum_{t < T} train_count(c, t)

    This matches best_accum(T), which uses all ckps < T together.
    """
    if cid not in train_counts:
        return 0

    raw = 0
    for t_train, count in train_counts[cid].items():
        if t_train < T:
            raw += count
    return raw


def compute_total_train_before_T(
    T: int,
    train_counts: Dict[int, Dict[int, int]],
) -> int:
    """
    Total number of train images across all classes and checkpoints < T:

        total_train_before_T(T) = sum_{c'} sum_{t < T} train_count(c', t)
    """
    total = 0
    for per_ckp in train_counts.values():
        for t_train, count in per_ckp.items():
            if t_train < T:
                total += count
    return total


def assign_exposure_bin(val: float) -> int:
    """
    Assign past_exposure RATIO value to bin index 0..4 based on EXPOSURE_BINS.
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
    test_path = json_dir / "test.json"
    rare_path = json_dir / "rare.json"

    print(f"\n=== Dataset: {dataset} ===")
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
    #   - mean_past_ratio_rare[T]
    #   - mean_past_ratio_all[T] (test + rare)
    #   - bin_counts_rare[T][bin_idx]
    #   - bin_counts_all[T][bin_idx]
    mean_past_ratio_rare: Dict[int, float] = {}
    mean_past_ratio_all: Dict[int, float] = {}

    bin_counts_rare: Dict[int, List[int]] = {}
    bin_counts_all: Dict[int, List[int]] = {}

    total_rare_per_ckp: Dict[int, int] = {}
    total_eval_per_ckp: Dict[int, int] = {}

    for ckp_name in ckps:
        T = ckp_name_to_index(ckp_name)
        rare_entries = rare_data.get(ckp_name, [])
        test_entries = test_data.get(ckp_name, [])

        # denominator: how much training has Accum seen before T?
        total_train_before_T = compute_total_train_before_T(T, train_counts)

        # ----- Rare-only exposure ratios -----
        ratio_values_rare = []
        bin_count_rare = [0] * len(BIN_LABELS)

        for e in rare_entries:
            cid = int(e["class_id"])
            past_exp = compute_past_exposure(cid, T, train_counts)  # counts
            if total_train_before_T > 0:
                ratio = past_exp / float(total_train_before_T)
            else:
                ratio = 0.0

            ratio_values_rare.append(ratio)

            bidx = assign_exposure_bin(ratio)
            bin_count_rare[bidx] += 1

        mean_ratio_rare = sum(ratio_values_rare) / len(ratio_values_rare) if ratio_values_rare else 0.0
        mean_past_ratio_rare[T] = mean_ratio_rare
        bin_counts_rare[T] = bin_count_rare
        total_rare_per_ckp[T] = len(rare_entries)

        # ----- Combined (TEST + RARE) exposure ratios -----
        all_entries = list(test_entries) + list(rare_entries)
        ratio_values_all = []
        bin_count_all = [0] * len(BIN_LABELS)

        for e in all_entries:
            cid = int(e["class_id"])
            past_exp = compute_past_exposure(cid, T, train_counts)
            if total_train_before_T > 0:
                ratio = past_exp / float(total_train_before_T)
            else:
                ratio = 0.0

            ratio_values_all.append(ratio)

            bidx = assign_exposure_bin(ratio)
            bin_count_all[bidx] += 1

        mean_ratio_all = sum(ratio_values_all) / len(ratio_values_all) if ratio_values_all else 0.0
        mean_past_ratio_all[T] = mean_ratio_all
        bin_counts_all[T] = bin_count_all
        total_eval_per_ckp[T] = len(all_entries)

    # Prepare x-axis checkpoints (T values)
    Ts = sorted(mean_past_ratio_rare.keys())

    # Build lists for plotting
    mean_past_ratio_rare_list = [mean_past_ratio_rare[T] for T in Ts]
    mean_past_ratio_all_list = [mean_past_ratio_all[T] for T in Ts]

    mean_past_ratio_rare_pct = [r * 100.0 for r in mean_past_ratio_rare_list]
    mean_past_ratio_all_pct = [r * 100.0 for r in mean_past_ratio_all_list]

    # Rare performance (aligned with Ts; use NaN if missing)
    zs_vals = [zs_curve.get(T, math.nan) for T in Ts]
    oracle_vals = [oracle_curve.get(T, math.nan) for T in Ts]
    accum_vals = [accum_curve.get(T, math.nan) for T in Ts]

    # Convert bin_counts_rare to fractions per checkpoint
    bin_fracs_rare = {b: [] for b in range(len(BIN_LABELS))}  # bin_idx -> list over Ts
    for T in Ts:
        total = total_rare_per_ckp[T]
        counts = bin_counts_rare[T]
        if total > 0:
            fracs = [c / total for c in counts]
        else:
            fracs = [0.0] * len(BIN_LABELS)
        for b in range(len(BIN_LABELS)):
            bin_fracs_rare[b].append(fracs[b])

    # Convert bin_counts_all to fractions per checkpoint
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
    fig_path = FIG_OUT_DIR / f"exposure_quality_ratio_{tag}.png"

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.6, 1.6]}
    )

    # ----- Top: rare performance (left-y) + mean past exposure ratios (right-y) -----
    ax_perf = ax1
    ax_exp = ax1.twinx()

    ax_perf.plot(Ts, zs_vals, marker="o", linestyle="-", label="rare ZS")
    ax_perf.plot(Ts, oracle_vals, marker="s", linestyle="-", label="rare Best Oracle")
    ax_perf.plot(Ts, accum_vals, marker="^", linestyle="-", label="rare Best Accum")

    ax_perf.set_ylabel(f"Rare {RARE_METRIC_KEY}", fontsize=11)
    ax_perf.set_ylim(0.0, 1.0)
    ax_perf.grid(True, axis="y", linestyle="--", alpha=0.4)

    RARE_COLOR = "#D62728"      # red (matplotlib default red)
    RARETEST_COLOR = "#9467BD"  # purple (matplotlib default purple)


    ax_exp.plot(
        Ts,
        mean_past_ratio_rare_pct,
        marker="d",
        linestyle="--",
        label="Mean past exposure ratio (rare)",
        alpha=0.7,
        color=RARE_COLOR,
    )
    ax_exp.plot(
        Ts,
        mean_past_ratio_all_pct,
        marker="^",
        linestyle="--",
        label="Mean past exposure ratio (test + rare)",
        alpha=0.7,
        color=RARETEST_COLOR,
    )
    ax_exp.set_ylabel("Mean past exposure ratio\n(% of train before T)", fontsize=11)
    ax_exp.set_ylim(0, 50)


    ax_perf.set_title(f"Past Exposure Ratio & Rare Performance for {dataset}", fontsize=13)

    lines_perf, labels_perf = ax_perf.get_legend_handles_labels()
    lines_exp, labels_exp = ax_exp.get_legend_handles_labels()
    ax_perf.legend(
        lines_perf + lines_exp,
        labels_perf + labels_exp,
        loc="upper right",
        fontsize=9,
    )

    # ----- Middle: stacked bar of past exposure ratio bins (RARE only) -----
    bottoms = [0.0] * len(Ts)
    for bidx, label in enumerate(BIN_LABELS):
        vals = bin_fracs_rare[bidx]
        ax2.bar(Ts, vals, bottom=bottoms, label=label)
        bottoms = [bottoms[i] + vals[i] for i in range(len(Ts))]

    ax2.set_ylabel("Fraction of rare images", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(
        title="Past exposure ratio bin (rare)\n(past_exposure / total_train_before_T)",
        fontsize=9,
        title_fontsize=10,
        ncol=3,
        loc="upper right",
    )

    # ----- Bottom: stacked bar of past exposure ratio bins (TEST + RARE) -----
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
        title="Past exposure ratio bin (test + rare)\n(past_exposure / total_train_before_T)",
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
