#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
base_dir = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_feature_saving_new_threshold"
pattern = "*_per_ckp.csv"

# outputs (only these two artifacts)
nonzero_csv = os.path.join("always_no_pick_higher_nonzero_diffs_feature_saving_new_threshold0.15.csv")
hist_png    = os.path.join("always_no_pick_higher_nonzero_hist_015_feature_saving_new_threshold0.15.png")

# ====== LOAD & COLLECT ======
rows = []  # will hold dicts: dataset, ckp, diff

csv_files = sorted(glob(os.path.join(base_dir, pattern)))
if not csv_files:
    raise FileNotFoundError(f"No CSV files matched {os.path.join(base_dir, pattern)}")

for path in csv_files:
    try:
        df = pd.read_csv(path)
        dataset = os.path.basename(path).replace("_per_ckp.csv", "")

        ay = pd.to_numeric(df.get("always_no"), errors="coerce")
        ph = pd.to_numeric(df.get("pick_higher"), errors="coerce")
        if ay is None or ph is None:
            print(f"Skipping (missing columns): {path}")
            continue

        diffs = (ay - ph).abs()
        ckps = df.get("ckp")
        if ckps is None:
            ckps = [f"row_{i}" for i in range(len(df))]

        for ckp_label, d in zip(ckps, diffs):
            if pd.isna(d):
                continue
            rows.append({
                "dataset": dataset,
                "ckp": str(ckp_label),
                "diff": float(d),
            })

    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")

if not rows:
    raise RuntimeError("No valid rows collected — check data/column names.")

all_df = pd.DataFrame(rows)

# ====== FILTER OUT ZEROS ======
eps = 1e-12
nonzero_df = all_df[all_df["diff"] > 0.15 + eps].copy()
zero_df = all_df[(all_df["diff"] <= 0.15 + eps) & (all_df["diff"] >= -eps)].copy()
print(f"Total rows collected: {len(all_df)}")
print(f"Rows with non-zero diffs: {len(nonzero_df)}")
print(f"Rows filtered out (zero diffs): {len(zero_df)}")

# Print number of datasets left (with at least one non-zero diff)
datasets_left = nonzero_df["dataset"].nunique()
total_datasets = all_df["dataset"].nunique()
print(f"✅ Non-zero diffs found in {datasets_left} datasets out of {total_datasets} total.")

# ====== SAVE CSV (dataset, ckp, diff) ======
nonzero_df.to_csv(nonzero_csv, index=False)
print(f"Saved non-zero diffs to: {nonzero_csv} (rows={len(nonzero_df)})")

# ====== HISTOGRAM (0.05 bins, exact counts) ======
all_diffs = nonzero_df["diff"].to_numpy()
if all_diffs.size == 0:
    raise RuntimeError("No non-zero diffs to plot.")

bin_width = 0.05
bins = np.arange(0, 1.0000001, bin_width)   # 0.00, 0.05, ..., 1.00

plt.figure(figsize=(12, 4.5))
plt.hist(all_diffs, bins=bins, density=False)  # counts, not density
plt.xlabel("|always_no - pick_higher| (zeros removed)")
plt.ylabel("Count")
plt.title("Histogram of Absolute Differences Across All Datasets")
plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.0000001, 0.05))
plt.tight_layout()
plt.savefig(hist_png, dpi=150)
plt.close()
print(f"Saved histogram to: {hist_png}")
