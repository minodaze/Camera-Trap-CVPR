#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute ZS average (first row), BA average (diagonal), Best-model average (per-column max),
and GAP = Best - max(BA, ZS) for a list of datasets.

Assumptions:
- Each CSV path is: {ROOT}/{dataset_path_with_slash_replaced_by_underscore}_matrix_balanced_accuracy.csv
  e.g., "APN/APN_13U" -> ".../APN_APN_13U_matrix_balanced_accuracy.csv"
- First column contains model names: 'zs', 'model_2', ..., 'model_N'
- Columns are 'ckp_1', 'ckp_2', ..., 'ckp_K'
"""

from pathlib import Path
import math
import pandas as pd

# ================== CONFIG: EDIT THESE ==================
ROOT = Path("/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only")

DATASETS = [
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
# ========================================================


def dataset_to_csv_path(root: Path, dataset: str) -> Path:
    base = dataset.replace("/", "_")
    return root / f"{base}_matrix_balanced_accuracy.csv"


def numeric_part_after_prefix(name: str, prefix: str) -> int | None:
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name.startswith(prefix):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def compute_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Returns (ba_avg, zs_avg, best_avg, gap)
    - zs_avg: mean of row 'zs' across ckp_* columns
    - ba_avg: mean of diagonal: (zs, ckp_1), (model_2, ckp_2), ..., (model_K, ckp_K) if present
    - best_avg: mean over columns of the max across all rows
    - gap: best_avg - max(zs_avg, ba_avg)
    """
    # Standardize columns/index
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns[0].lower().startswith("model"):
        df = df.rename(columns={df.columns[0]: "model"})
    # set model column as index
    if df.columns[0] != "model":
        # ensure first column is treated as model names
        df = df.rename(columns={df.columns[0]: "model"})
    df["model"] = df["model"].astype(str).str.strip()
    df = df.set_index("model", drop=True)

    # select checkpoint columns, ordered by numeric suffix
    ckp_cols = []
    for c in df.columns:
        n = numeric_part_after_prefix(c, "ckp")
        if n is None:
            n = numeric_part_after_prefix(c, "ckp_")
        if n is None and c.startswith("ckp_"):
            # fallback parse after 'ckp_'
            try:
                n = int(c.split("_")[1])
            except Exception:
                n = None
        if n is not None:
            ckp_cols.append((n, c))
    ckp_cols.sort(key=lambda x: x[0])
    ordered_cols = [c for _, c in ckp_cols]
    if not ordered_cols:
        raise ValueError("No checkpoint columns (ckp_*) found.")

    # Make values numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # ---- ZS average
    if "zs" not in df.index:
        raise ValueError("Row 'zs' not found in the CSV.")
    zs_series = df.loc["zs", ordered_cols]
    zs_avg = zs_series.mean(skipna=True)

    # ---- BA average (diagonal)
    # diagonal mapping: (row, col): (zs, ckp_1), (model_2, ckp_2), ..., (model_K, ckp_K)
    ba_vals = []
    for i, col in enumerate(ordered_cols, start=1):
        row_name = "zs" if i == 1 else f"model_{i}"
        if row_name in df.index and col in df.columns:
            val = df.at[row_name, col]
            if pd.notna(val):
                ba_vals.append(val)
    if not ba_vals:
        raise ValueError("No diagonal BA values found (check model_x rows and ckp_* columns).")
    ba_avg = float(sum(ba_vals) / len(ba_vals))

    # ---- Best-model average (per-column max over all rows)
    best_per_col = []
    for col in ordered_cols:
        col_max = df[col].max(skipna=True)
        if pd.notna(col_max):
            best_per_col.append(col_max)
    if not best_per_col:
        raise ValueError("No per-column maxima found.")
    best_avg = float(sum(best_per_col) / len(best_per_col))

    # ---- GAP
    gap = best_avg - max(zs_avg, ba_avg)

    return float(ba_avg), float(zs_avg), float(best_avg), float(gap)


def main():
    rows = []
    missing = []

    for ds in DATASETS:
        csv_path = dataset_to_csv_path(ROOT, ds)
        if not csv_path.exists():
            missing.append((ds, str(csv_path)))
            continue

        try:
            # read CSV; allow odd formatting
            df = pd.read_csv(csv_path, engine="python")
            ba_avg, zs_avg, best_avg, gap = compute_metrics(df)
            rows.append({
                "dataset": ds.replace("/", "_"),
                "BA_avg": round(ba_avg, 6),
                "ZS_avg": round(zs_avg, 6),
                "Best_avg": round(best_avg, 6),
                "GAP": round(gap, 6),
            })
        except Exception as e:
            rows.append({
                "dataset": ds.replace("/", "_"),
                "BA_avg": math.nan,
                "ZS_avg": math.nan,
                "Best_avg": math.nan,
                "GAP": math.nan,
                "error": str(e),
            })

    summary = pd.DataFrame(rows, columns=["dataset", "BA_avg", "ZS_avg", "Best_avg", "GAP", "error"])
    out_path = Path("summary.csv")
    summary.to_csv(out_path, index=False)

    print("\n=== Summary written to:", out_path.resolve(), "===\n")
    print(summary.fillna(""))

    if missing:
        print("\nMissing files:")
        for ds, path in missing:
            print(f" - {ds}: {path}")


if __name__ == "__main__":
    main()
