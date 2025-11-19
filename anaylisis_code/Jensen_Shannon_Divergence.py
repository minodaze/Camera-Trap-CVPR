#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from typing import List

# ================== CONFIG ==================
# Roots that match your current layout
COUNTS_ROOT = "/users/PAS2099/lemengwang/Documents/icicle/Camera-Trap-CVPR/class_counts"
OUT_ROOT    = "/users/PAS2099/lemengwang/Documents/icicle/Camera-Trap-CVPR/class_distribution"

# Datasets to process
DATASETS: List[str] = [
    "nz/nz_EFD_DCAME02",
    "serengeti/serengeti_E08",
    "nz/nz_EFD_DCAMB02",
    "nz/nz_PS1_CAM7709",
    "nz/nz_EFH_HCAMI01",
    "nz/nz_HLO_Moto2",
    "nz/nz_EFH_HCAMH03",
    "nz/nz_EFH_HCAMC04",
    "nz/nz_EFH_HCAMB04",
    "na/na_lebec_CA-36",
]

# Which split file to read: "train_counts.csv" (change to val/test if needed)
SPLIT_FILENAME = "test_counts.csv"
# Output filename (kept consistent with your example)
OUT_FILENAME   = "ckp_shift_metrics_test.csv"

# ================== METRICS ==================

def _safe_dist(vec, eps=1e-12):
    v = np.asarray(vec, dtype=float) + eps
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / len(v)
    return v / s

def gini(p):
    p = _safe_dist(p)
    return 1.0 - np.sum(p**2)

def tv_distance(p, q):
    p = _safe_dist(p); q = _safe_dist(q)
    return 0.5 * np.abs(p - q).sum()

def jsd(p, q, base=2):
    p = _safe_dist(p); q = _safe_dist(q)
    m = 0.5 * (p + q)
    def _kl(a, b):
        return np.sum(a * (np.log(a) - np.log(b))) / np.log(base)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def hellinger(p, q):
    p = _safe_dist(p); q = _safe_dist(q)
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)

# ================== CORE LOGIC ==================

def compute_shift_metrics(df: pd.DataFrame, ckp_col: str = "ckp") -> pd.DataFrame:
    # Ensure numeric sorting by ckp
    if not np.issubdtype(df[ckp_col].dtype, np.number):
        df = df.copy()
        df[ckp_col] = pd.to_numeric(df[ckp_col], errors="coerce")
    df = df.sort_values(ckp_col).reset_index(drop=True)

    # Identify class columns (drop ckp and optional TOTAL)
    class_cols = [c for c in df.columns if c not in {ckp_col, "TOTAL"}]
    counts = df[class_cols].to_numpy(dtype=float)
    ckps = df[ckp_col].to_list()

    results = []
    cum_prev = np.zeros(counts.shape[1], dtype=float)

    for i in range(len(df)):
        curr = counts[i]
        gini_curr = gini(curr)

        # vs cumulative previous
        if cum_prev.sum() == 0:
            jsd_cum = np.nan; hel_cum = np.nan; tv_cum = np.nan; gini_cum = np.nan
        else:
            jsd_cum = jsd(curr, cum_prev, base=2)
            hel_cum = hellinger(curr, cum_prev)
            tv_cum  = tv_distance(curr, cum_prev)
            gini_cum = gini(cum_prev)

        # vs immediate previous
        if i == 0:
            jsd_prev = np.nan; hel_prev = np.nan; tv_prev = np.nan
        else:
            prev = counts[i - 1]
            jsd_prev = jsd(curr, prev, base=2)
            hel_prev = hellinger(curr, prev)
            tv_prev  = tv_distance(curr, prev)

        results.append({
            ckp_col: ckps[i],
            "Gini_current": gini_curr,
            "JSD_vs_cumulative(0-1)": jsd_cum,
            "Hellinger_vs_cumulative(0-1)": hel_cum,
            "TV_vs_cumulative(0-1)": tv_cum,
            "Gini_cumulative": gini_cum,
            "JSD_vs_prev(0-1)": jsd_prev,
            "Hellinger_vs_prev(0-1)": hel_prev,
            "TV_vs_prev(0-1)": tv_prev,
        })

        cum_prev += curr

    return pd.DataFrame(results)

def ensure_parent_dir(path: str):
    """Create parent directory for a file path, handle directory/file collisions."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
        print(f"Created directory: {parent}")
    # If a directory already exists with the SAME name as 'path', fix by writing inside it
    if os.path.isdir(path):
        # Convert it to a file inside that directory
        fixed = os.path.join(path, "output.csv")
        print(f"[Warning] '{path}' is a directory; writing to '{fixed}' instead.")
        return fixed
    return path

def dataset_name_from_key(key: str) -> str:
    """Turn 'nz/nz_EFD_DCAME02' into 'nz_EFD_DCAME02' for folder names."""
    return key.split("/")[-1]

def process_dataset(ds_key: str):
    ds_name = dataset_name_from_key(ds_key)

    csv_path = os.path.join(COUNTS_ROOT, ds_name, SPLIT_FILENAME)
    out_dir  = os.path.join(OUT_ROOT, ds_name)
    out_path = os.path.join(out_dir, OUT_FILENAME)

    if not os.path.exists(csv_path):
        print(f"[Skip] Missing CSV for {ds_key}: {csv_path}")
        return

    # Read CSV and compute
    df = pd.read_csv(csv_path)
    if "ckp" not in df.columns:
        print(f"[Skip] No 'ckp' column in {csv_path}")
        return

    metrics = compute_shift_metrics(df, ckp_col="ckp")

    # Prepare output path, handle collisions
    out_path = ensure_parent_dir(out_path)

    # Save & print head
    pd.set_option("display.precision", 6)
    print(f"\n=== {ds_key} ===")
    print(metrics.head().to_string(index=False))
    metrics.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

def main():
    for ds in DATASETS:
        process_dataset(ds)

if __name__ == "__main__":
    main()
