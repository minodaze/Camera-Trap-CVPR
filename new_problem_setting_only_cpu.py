#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU-only resweep using dataset-specific baseline thresholds from T_01.

For each dataset:
  - Feature threshold (tau_feat): mean(per_target_dists) at diagnostics/<DS_U>/T_01/
  - MSP threshold (tau_msp): fixed by default, or mean(msp_scores) at T_01 if --msp_from_t1 is set

Reads (already produced by the GPU run):
  diagnostics/<DS_U>/T_xx/msp_scores.npy
  diagnostics/<DS_U>/T_xx/per_target_dists.npy
  acc_matrices_from_eval_only/<DS_U>_matrix_balanced_accuracy.csv

Writes per-dataset:
  when_to_adapt_out/<DS_U>/summary_resweep_t1.csv   (default suffix 'resweep_t1')
  (optional) NP_RESULT_DIR/<DS_U>_per_ckp_resweep_t1.csv
"""

import os, os.path as osp, argparse, sys, math
from typing import List, Optional
import numpy as np
import pandas as pd

# ====== PATHS (edit if needed) ===============================================
MATRIX_DIR    = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only"
SUMMARY_ROOT  = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_feature_saving/threshold_from_t1_for_both"
DIAG_DIR      = osp.join(SUMMARY_ROOT, "diagnostics")
NP_RESULT_DIR = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_feature_saving/threshold_from_t1_for_both"

# ====== DEFAULTS ==============================================================

# If you don't pass datasets via CLI we fall back to this list.
DATASETS = [
    "ENO/ENO_D06","serengeti/serengeti_N04","serengeti/serengeti_H11","APN/APN_K051",
    "nz/nz_EFH_HCAME05","na/na_lebec_CA-21","MAD/MAD_A04","MTZ/MTZ_F04","APN/APN_13U",
    "nz/nz_EFD_DCAMG03","nz/nz_PS1_CAM6213","nz/nz_EFH_HCAME09","MAD/MAD_D04",
    "wellington/wellington_031c","KAR/KAR_A01","na/na_lebec_CA-18","serengeti/serengeti_Q09",
    "APN/APN_U43B","APN/APN_K082","APN/APN_N1","nz/nz_EFH_HCAME08","serengeti/serengeti_L06",
    "nz/nz_EFH_HCAMF01","caltech/caltech_46","serengeti/serengeti_S11","serengeti/serengeti_O13",
    "PLN/PLN_B04","MTZ/MTZ_E05","nz/nz_EFD_DCAMH07","caltech/caltech_70","nz/nz_EFH_HCAMB05",
    "KAR/KAR_B03","serengeti/serengeti_D02","MAD/MAD_B03","nz/nz_EFD_DCAMF06","caltech/caltech_88",
    "na/na_lebec_CA-19","APN/APN_U23A","na/na_lebec_CA-05","nz/nz_EFH_HCAMI01","KGA/KGA_KHOGA04",
    "ENO/ENO_C02","ENO/ENO_C04","MAD/MAD_C07","serengeti/serengeti_E05","serengeti/serengeti_V10",
    "na/na_lebec_CA-31","serengeti/serengeti_F08","MAD/MAD_B6","nz/nz_EFH_HCAMB01",
    "KGA/KGA_KHOLA03","nz/nz_EFH_HCAMD08","nz/nz_EFH_HCAMC03","serengeti/serengeti_L10",
    "serengeti/serengeti_D09","idaho/idaho_122","serengeti/serengeti_Q11","MAD/MAD_H08",
    "CDB/CDB_A05","serengeti/serengeti_E12","nz/nz_EFH_HCAMC02","serengeti/serengeti_T10",
    "serengeti/serengeti_H03","nz/nz_PS1_CAM8008","na/na_lebec_CA-37","serengeti/serengeti_R10",
    "nz/nz_EFD_DCAMH01","nz/nz_EFH_HCAMG13","serengeti/serengeti_K11","APN/APN_WM",
    "nz/nz_PS1_CAM7312","ENO/ENO_E06","serengeti/serengeti_Q10","serengeti/serengeti_H08",
    "APN/APN_TB17","serengeti/serengeti_Q07","caltech/caltech_38","MTZ/MTZ_D06",
    "nz/nz_EFD_DCAMD10","MTZ/MTZ_D03",
]

# ====== HELPERS ==============================================================

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _d2u(ds: str) -> str: return ds.replace("/", "_")
def _ckp_name(T: int) -> str: return f"ckp_{T}"
def _row_name_for_model_k(k: int) -> str: return "zs" if k == 1 else f"model_{k}"

def _load_matrix(ds: str) -> pd.DataFrame:
    ds_u = _d2u(ds)
    f = osp.join(MATRIX_DIR, f"{ds_u}_matrix_balanced_accuracy.csv")
    df = pd.read_csv(f)
    first = df.columns[0]
    df = df.rename(columns={first: "model"})
    df["model"] = df["model"].astype(str).str.strip()
    df = df.set_index("model")
    ckp_cols = [c for c in df.columns if c.startswith("ckp_")]
    ckp_cols.sort(key=lambda c: int(c.split("_")[1]))
    return df[ckp_cols].astype(float)

def _acc(df: pd.DataFrame, model_k: int, T: int) -> float:
    row = _row_name_for_model_k(model_k)
    col = _ckp_name(T)
    if row not in df.index or col not in df.columns: return float("nan")
    v = df.loc[row, col]
    return float(v) if not pd.isna(v) else float("nan")

def _available_Ts(df: pd.DataFrame):
    Ts = []
    for c in df.columns:
        if c.startswith("ckp_"):
            try: Ts.append(int(c.split("_")[1]))
            except: pass
    return sorted(Ts)

def _diag_dir(ds_u: str, T: int) -> str:
    return osp.join(DIAG_DIR, ds_u, f"T_{T:02d}")

def _load_mean(fp: str) -> Optional[float]:
    if not osp.exists(fp): return None
    x = np.load(fp).astype(np.float32)
    return float(x.mean()) if x.size else None

def _t1_feat_tau(ds_u: str) -> Optional[float]:
    """tau_feat = mean per-target distance at T_01 for this dataset."""
    return _load_mean(osp.join(_diag_dir(ds_u, 1), "per_target_dists.npy"))

def _t1_msp_tau(ds_u: str) -> Optional[float]:
    """tau_msp = mean MSP at T_01 for this dataset."""
    return _load_mean(osp.join(_diag_dir(ds_u, 1), "msp_scores.npy"))

# ====== CORE RESWEEP =========================================================

def resweep_dataset(
    ds: str,
    tau_msp_fixed: float,
    use_msp_from_t1: bool,
    out_suffix: str = "resweep_t1",
    write_flat: bool = True,
):
    ds_u = _d2u(ds)
    dfM  = _load_matrix(ds)
    Ts   = _available_Ts(dfM)

    # Get dataset-specific thresholds
    tau_feat = _t1_feat_tau(ds_u)
    if tau_feat is None or math.isnan(tau_feat):
        print(f"[WARN] {ds}: missing T_01 per_target_dists.npy; class_mean will be NaN.", file=sys.stderr)

    if use_msp_from_t1:
        tau_msp = _t1_msp_tau(ds_u)
        if tau_msp is None or math.isnan(tau_msp):
            print(f"[WARN] {ds}: missing T_01 msp_scores.npy; MSP uses fixed={tau_msp_fixed}.", file=sys.stderr)
            tau_msp = tau_msp_fixed
    else:
        tau_msp = tau_msp_fixed

    rows = []
    for T in Ts:
        prev_k = max(1, T-1)
        this_k = T

        # Always- rules
        a_yes = _acc(dfM, this_k, T)
        a_no  = _acc(dfM, prev_k, T)

        # MSP decision at T: compare mean MSP(T) vs tau_msp
        msp_fp = osp.join(_diag_dir(ds_u, T), "msp_scores.npy")
        msp_mean = _load_mean(msp_fp)
        if msp_mean is None or math.isnan(msp_mean):
            a_msp = float("nan")
        else:
            yes_msp = (msp_mean < tau_msp)
            a_msp = _acc(dfM, this_k if yes_msp else prev_k, T)

        # Feature decision at T: compare mean dist(T) vs tau_feat_from_T01
        dist_fp = osp.join(_diag_dir(ds_u, T), "per_target_dists.npy")
        dist_mean = _load_mean(dist_fp)
        if (tau_feat is None) or math.isnan(tau_feat) or (dist_mean is None) or math.isnan(dist_mean):
            a_feat = float("nan")
        else:
            yes_feat = (dist_mean > tau_feat)
            a_feat = _acc(dfM, this_k if yes_feat else prev_k, T)

        # Oracle
        if math.isnan(a_yes) and math.isnan(a_no):
            a_pick = float("nan")
        elif math.isnan(a_yes):
            a_pick = a_no
        elif math.isnan(a_no):
            a_pick = a_yes
        else:
            a_pick = max(a_yes, a_no)

        rows.append({
            "ckp": _ckp_name(T),
            "random": float("nan"),
            "always_yes": a_yes,
            "always_no":  a_no,
            "ood_msp":    a_msp,
            "class_mean": a_feat,
            "pick_higher": a_pick,
        })

    df_out = pd.DataFrame(rows, columns=["ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"])

    # Write per-dataset summary and optional flat file
    ds_folder = osp.join(SUMMARY_ROOT, ds_u)
    _ensure_dir(ds_folder)
    summary_path = osp.join(ds_folder, f"summary_{out_suffix}.csv")
    df_out.to_csv(summary_path, index=False)
    print(f"💾 [{ds}] wrote {summary_path}")

    if write_flat:
        _ensure_dir(NP_RESULT_DIR)
        flat_path = osp.join(NP_RESULT_DIR, f"{ds_u}_per_ckp_{out_suffix}.csv")
        df_out.to_csv(flat_path, index=False)
        print(f"💾 [{ds}] wrote {flat_path}")

# ====== CLI ==================================================================

def parse_cli():
    ap = argparse.ArgumentParser(description="CPU resweep using T_01 averages as thresholds.")
    ap.add_argument("--datasets", type=str, default="", help="Comma-separated datasets (slash-form).")
    ap.add_argument("--datasets_file", type=str, default="", help="File with one dataset per line.")
    ap.add_argument("--tau_msp_fixed", type=float, default=0.65, help="Fallback MSP threshold if not using/available from T_01.")
    ap.add_argument("--msp_from_t1", action="store_true", help="Use MSP mean at T_01 as tau_msp (else use --tau_msp_fixed).")
    ap.add_argument("--out_suffix", type=str, default="resweep_t1", help="Suffix for output CSV filenames.")
    ap.add_argument("--no_flat", action="store_true", help="Do not write the flat _per_ckp_ CSV.")
    return ap.parse_args()

def main():
    args = parse_cli()

    # Resolve datasets
    if args.datasets_file:
        with open(args.datasets_file, "r") as f:
            ds_list = [ln.strip() for ln in f if ln.strip()]
    elif args.datasets:
        ds_list = [x.strip() for x in args.datasets.split(",") if x.strip()]
    else:
        ds_list = DATASETS

    for ds in ds_list:
        try:
            resweep_dataset(
                ds=ds,
                tau_msp_fixed=args.tau_msp_fixed,
                use_msp_from_t1=args.msp_from_t1,
                out_suffix=args.out_suffix,
                write_flat=(not args.no_flat),
            )
        except Exception as e:
            print(f"[FAIL] {ds}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
