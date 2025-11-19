#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import numpy as np
from typing import List, Tuple

# ----------------- CONFIG (edit here) -----------------
DATASETS: List[str] = [
"ENO/ENO_D06",
"serengeti/serengeti_N04",
"serengeti/serengeti_H11",
"APN/APN_K051",
"nz/nz_EFH_HCAME05",
"na/na_lebec_CA-21",
"MAD/MAD_A04",
"MTZ/MTZ_F04",
"APN/APN_13U",
"nz/nz_EFD_DCAMG03",
"nz/nz_PS1_CAM6213",
"nz/nz_EFH_HCAME09",
"MAD/MAD_D04",
"wellington/wellington_031c",
"KAR/KAR_A01",
"na/na_lebec_CA-18",
"serengeti/serengeti_Q09",
"APN/APN_U43B",
"APN/APN_K082",
"APN/APN_N1",
"nz/nz_EFH_HCAME08",
"serengeti/serengeti_L06",
"nz/nz_EFH_HCAMF01",
"caltech/caltech_46",
"serengeti/serengeti_S11",
"serengeti/serengeti_O13",
"PLN/PLN_B04",
"MTZ/MTZ_E05",
"nz/nz_EFD_DCAMH07",
"caltech/caltech_70",
"nz/nz_EFH_HCAMB05",
"KAR/KAR_B03",
"serengeti/serengeti_D02",
"MAD/MAD_B03",
"nz/nz_EFD_DCAMF06",
"caltech/caltech_88",
"na/na_lebec_CA-19",
"APN/APN_U23A",
"na/na_lebec_CA-05",
"nz/nz_EFH_HCAMI01",
"KGA/KGA_KHOGA04",
"ENO/ENO_C02",
"ENO/ENO_C04",
"MAD/MAD_C07",
"serengeti/serengeti_E05",
"serengeti/serengeti_V10",
"na/na_lebec_CA-31",
"serengeti/serengeti_F08",
"MAD/MAD_B06",
"nz/nz_EFH_HCAMB01",
"KGA/KGA_KHOLA03",
"nz/nz_EFH_HCAMD08",
"nz/nz_EFH_HCAMC03",
"serengeti/serengeti_L10",
"serengeti/serengeti_D09",
"idaho/idaho_122",
"serengeti/serengeti_Q11",
"MAD/MAD_H08",
"CDB/CDB_A05",
"serengeti/serengeti_E12",
"nz/nz_EFH_HCAMC02",
"serengeti/serengeti_T10",
"serengeti/serengeti_H03",
"nz/nz_PS1_CAM8008",
"na/na_lebec_CA-37",
"serengeti/serengeti_R10",
"nz/nz_EFD_DCAMH01",
"nz/nz_EFH_HCAMG13",
"serengeti/serengeti_K11",
"APN/APN_WM",
"nz/nz_PS1_CAM7312",
"ENO/ENO_E06",
"serengeti/serengeti_Q10",
"serengeti/serengeti_H08",
"APN/APN_TB17",
"serengeti/serengeti_Q07",
"caltech/caltech_38",
"MTZ/MTZ_D06",
"nz/nz_EFD_DCAMD10",
"MTZ/MTZ_D03",
]

# Where eval_only_summary.json is located for each dataset
EVAL_LOG_ROOT = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/test_logs"
EVAL_JSON_TAIL = "best_accum/bioclip2/lora_8_text_head_merge_factor_1/all/log/eval_only_summary.json"

# Where to write matrices
OUT_ROOT = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only"

# Which metric to pull from the JSON entries
METRIC_KEY = "balanced_accuracy"   # or "accuracy"
# ------------------------------------------------------


def group_dataset_to_combo(key: str) -> Tuple[str, str, str]:
    if "/" not in key:
        raise ValueError(f"Bad dataset key '{key}', expected 'GROUP/DATASET'")
    g, d = key.split("/", 1)
    return g, d, f"{g}_{d}"


def load_eval_json(combo: str) -> dict:
    path = os.path.join(EVAL_LOG_ROOT, combo, EVAL_JSON_TAIL)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"eval_only_summary.json not found for {combo}: {path}")
    with open(path, "r") as f:
        return json.load(f)


def extract_ckp_numbers(block_keys: List[str]) -> List[int]:
    # keep only "ckp_<int>"
    out = []
    for k in block_keys:
        if k.startswith("ckp_"):
            try:
                out.append(int(k.split("_")[1]))
            except Exception:
                pass
    return sorted(out)


def get_metric(cell: dict, metric_key: str) -> float:
    if cell is None:
        return math.nan
    val = cell.get(metric_key, None)
    if val is None:
        return math.nan
    try:
        return float(val)
    except Exception:
        return math.nan


def build_matrix(eval_summary: dict, metric_key: str):
    """
    eval_summary structure (accu_eval):
      {
        "ckp_1": { "ckp_1": {...}, "ckp_2": {...}, ... },
        "ckp_2": { "ckp_2": {...}, "ckp_3": {...}, ... },
        ...
        "average": {...}
      }
    """
    # All top-level ckps present (ignore "average")
    model_rows = [k for k in eval_summary.keys() if k.startswith("ckp_")]
    t_all = extract_ckp_numbers(model_rows)

    if not t_all:
        raise ValueError("No checkpoint blocks found in eval_summary")

    K = max(t_all)

    # Row tags: zs, model_2..model_K
    row_tags = ["zs"] + [f"model_{t}" for t in range(2, K + 1)]
    # Column tags: ckp_1..ckp_K
    col_tags = [f"ckp_{t}" for t in range(1, K + 1)]

    M = np.full((len(row_tags), len(col_tags)), np.nan, dtype=float)

    # ---- zs row is the "ckp_1" block evaluated on all columns it has
    zs_block = eval_summary.get("ckp_1", {})
    for j, c in enumerate(col_tags):
        if c in zs_block:
            M[0, j] = get_metric(zs_block[c], metric_key)

    # ---- trained rows: row for t uses block "ckp_t" and fills cols ckp_t..ckp_K
    for i, tag in enumerate(row_tags[1:], start=1):
        t = int(tag.split("_")[1])  # model_t
        block_name = f"ckp_{t}"
        block = eval_summary.get(block_name, {})
        # Fill only future+current columns
        for j, c in enumerate(col_tags):
            tj = int(c.split("_")[1])
            if tj < t:
                continue
            cell = block.get(c, None)
            M[i, j] = get_metric(cell, metric_key)

    return row_tags, col_tags, M


def write_csv(out_path: str, row_tags: List[str], col_tags: List[str], M: np.ndarray):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model \\ test_ckp"] + col_tags)
        for i, r in enumerate(row_tags):
            row = [r] + [("" if np.isnan(v) else f"{v:.6f}") for v in M[i]]
            w.writerow(row)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for key in DATASETS:
        try:
            g, d, combo = group_dataset_to_combo(key)
            eval_summary = load_eval_json(combo)
            row_tags, col_tags, M = build_matrix(eval_summary, METRIC_KEY)

            out_csv = os.path.join(OUT_ROOT, f"{combo}_matrix_{METRIC_KEY}.csv")
            write_csv(out_csv, row_tags, col_tags, M)
            print(f"[OK] {combo}: wrote -> {out_csv}")
        except Exception as e:
            print(f"[FAIL] {key}: {e}")


if __name__ == "__main__":
    main()
