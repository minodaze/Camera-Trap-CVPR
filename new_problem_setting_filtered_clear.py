#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate 5 "when to adapt" algorithms ONLY at selected checkpoints from an allow-list CSV.

Now writes ONE global, incremental CSV with per-ckp rows:
columns = ["dataset","ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"]

We still emit per-dataset per-ckp CSVs, but we DO NOT write per-dataset means anymore.
"""

import os
import os.path as osp
import math
import glob
import json
import random
import sys
import gc
from types import SimpleNamespace
from typing import Dict, List, Set, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm, trange

# --- import from your repo ---
from core import *
from run_pipeline import parse_args

# ====== EDIT THESE PATHS ======================================================
MATRIX_DIR       = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only"
CONFIG_DIR       = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/finalConfig_np"
NP_RESULT_DIR    = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_filtered_shouldnotadapt"
ALLOWLIST_CSV    = "/users/PAS2099/lemengwang/Documents/icicle/Camera-Trap-CVPR/anaylisis_code/always_yes_pick_higher_nonzero_diffs.csv"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds (can be swept later)
GLOBAL_TAU_MSP_DEFAULT = 0.65
TAU_FEAT  = 0.20   # Algo 5: train if class-mean distance > 0.20

# Your dataset list (slash-form)
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
    "na/na_lebec_CA-31","serengeti/serengeti_F08","MAD/MAD_B06","nz/nz_EFH_HCAMB01",
    "KGA/KGA_KHOLA03","nz/nz_EFH_HCAMD08","nz/nz_EFH_HCAMC03","serengeti/serengeti_L10",
    "serengeti/serengeti_D09","idaho/idaho_122","serengeti/serengeti_Q11","MAD/MAD_H08",
    "CDB/CDB_A05","serengeti/serengeti_E12",
    "nz/nz_EFH_HCAMC02","serengeti/serengeti_T10",
    "serengeti/serengeti_H03","nz/nz_PS1_CAM8008",
    "na/na_lebec_CA-37","serengeti/serengeti_R10",
    "nz/nz_EFD_DCAMH01","nz/nz_EFH_HCAMG13","serengeti/serengeti_K11","APN/APN_WM",
    "nz/nz_PS1_CAM7312","ENO/ENO_E06","serengeti/serengeti_Q10","serengeti/serengeti_H08",
    "APN/APN_TB17","serengeti/serengeti_Q07","caltech/caltech_38","MTZ/MTZ_D06",
    "nz/nz_EFD_DCAMD10","MTZ/MTZ_D03",
]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
yaml_loader = YAML(typ="safe")

def _d2u(dataset: str) -> str:
    """'APN/APN_13U' -> 'APN_APN_13U'."""
    return dataset.replace("/", "_")

def _ckp_name(T: int) -> str:
    return f"ckp_{T}"

def _T_from_ckp(ckp: str) -> Optional[int]:
    try:
        return int(str(ckp).split("_")[1])
    except Exception:
        return None

def _row_name_for_model_k(k: int) -> str:
    return "zs" if k == 1 else f"model_{k}"

def _acc_from_matrix(df: pd.DataFrame, model_k: int, T: int) -> float:
    row = _row_name_for_model_k(model_k)
    col = _ckp_name(T)
    if row not in df.index or col not in df.columns:
        return float("nan")
    v = df.loc[row, col]
    return float(v) if not pd.isna(v) else float("nan")

def load_matrix_csv(dataset: str) -> pd.DataFrame:
    csv_path = os.path.join(MATRIX_DIR, f"{_d2u(dataset)}_matrix_balanced_accuracy.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    first = df.columns[0]
    df = df.rename(columns={first: "model"})
    df["model"] = df["model"].astype(str).str.strip()
    df = df.set_index("model")
    ckp_cols = [c for c in df.columns if c.startswith("ckp_")]
    ckp_cols.sort(key=lambda c: int(c.split("_")[1]))
    return df[ckp_cols].astype(float)

def load_config(dataset: str) -> SimpleNamespace:
    cfg_path = os.path.join(CONFIG_DIR, f"{_d2u(dataset)}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    argv_backup = sys.argv[:]
    try:
        sys.argv = [argv_backup[0], "--c", cfg_path]
        args = parse_args()
    finally:
        sys.argv = argv_backup

    setattr(args, "device", getattr(args, "device", DEVICE))
    setattr(args, "seed", getattr(args, "seed", 9527))
    setattr(args, "label_type", getattr(args, "label_type", "common"))
    return args

def get_class_names_from_json(train_json: str, test_json: str, label_key: str="common") -> List[str]:
    with open(train_json, "r") as f:
        d = json.load(f)
    with open(test_json, "r") as f:
        d2 = json.load(f)
    d.update(d2)
    names = []
    for _, arr in d.items():
        for v in arr:
            name = v[label_key]
            if name not in names:
                names.append(name)
    return names

def build_datasets_and_model(dataset: str, T: int):
    args = load_config(dataset)
    cc   = args.common_config

    class_names = get_class_names_from_json(
        cc["train_data_config_path"],
        cc["eval_data_config_path"],
        label_key = "common" if getattr(args, "label_type", "common") in ["common", "common_name"] else "scientific"
    )

    clf = build_classifier(args, class_names, args.device).to(DEVICE).eval()

    # Try load M(T) weights if available
    ds_u = _d2u(dataset)
    search_roots = [
        os.path.join(args.log_path, ds_u),
        args.log_path,
    ]
    weight_path = None
    for root in search_roots:
        pat = os.path.join(root, "**", f"ckp_ckp_{T}_best_model.pth")
        g = glob.glob(pat, recursive=True)
        if g:
            weight_path = sorted(g)[-1]
            break
    if weight_path and os.path.exists(weight_path):
        try:
            state = torch.load(weight_path, map_location=DEVICE)
            current_state_dict = clf.state_dict()
            filtered_state_dict = {}
            for key, value in state.items():
                if key in current_state_dict:
                    if current_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    elif 'head' in key:
                        if key == 'head.weight':
                            m = min(current_state_dict[key].shape[0], value.shape[0])
                            current_state_dict[key][:m] = value[:m]
                        elif key == 'head.bias':
                            m = min(current_state_dict[key].shape[0], value.shape[0])
                            current_state_dict[key][:m] = value[:m]
            clf.load_state_dict(filtered_state_dict, strict=False)
            clf.to(args.device).eval()
        except Exception as e:
            print(f"[WARN] Failed to load weights ({weight_path}): {e}. Using ZS weights.")

    train_ds = CkpDataset(cc["train_data_config_path"], class_names, label_type=getattr(args, "label_type", "common"))
    eval_ds  = CkpDataset(cc["eval_data_config_path"],  class_names, is_train=False, label_type=args.label_type)

    eval_T = eval_ds.get_subset(is_train=False, ckp_list=_ckp_name(T))
    if T > 1:
        train_pool = train_ds.get_subset(is_train=True, ckp_list=_ckp_name(T - 1))
    else:
        train_pool = train_ds.get_subset(is_train=True, ckp_list=_ckp_name(1))

    bs_eval = int(cc.get("eval_batch_size", 256))
    dl_kw   = dict(batch_size=bs_eval, shuffle=False, num_workers=4)

    eval_loader_T = DataLoader(eval_T, **dl_kw)
    train_pool_loader_1_to_Tm1 = DataLoader(train_pool, **dl_kw)

    return clf, eval_loader_T, train_pool_loader_1_to_Tm1

# -----------------------------------------------------------------------------
# Measurements for Algo 4 & 5
# -----------------------------------------------------------------------------
@torch.no_grad()
def avg_msp(model: nn.Module, loader) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        if isinstance(batch, dict):
            x = batch.get("images") or batch.get("pixel_values") or batch.get("input")
        else:
            x = batch[0]
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        msp = probs.max(dim=-1).values
        tot += msp.sum().item()
        n   += msp.numel()
    return tot / max(n, 1)

class FeatureHead(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
    def forward(self, x):
        if hasattr(self.base, "forward_features"):
            feats = self.base.forward_features(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            feats = feats.float().view(feats.size(0), -1)
            return feats
        else:
            logits = self.base(x)
            return logits.float()

@torch.no_grad()
def class_mean_distance_ref_vs_target(feat_model: nn.Module, ref_loader, target_loader) -> float:
    """
    Unlabeled TARGET (per-target distances):
      - L2-normalize REF class means (prototypes).
      - L2-normalize each TARGET feature.
      - Assign each TARGET to nearest prototype (max dot on normalized vectors).
      - Distance = mean over TARGET of || target_norm - proto_norm ||_2.
    """
    feat_model.eval()

    def _extract_feats_and_labels(loader):
        xs, ys = [], []
        for batch in loader:
            if isinstance(batch, dict):
                x = batch.get("images") or batch.get("pixel_values") or batch.get("input")
                y = batch.get("labels") if "labels" in batch or "label" in batch else None
            else:
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            x = x.to(DEVICE, non_blocking=True)
            feats = feat_model(x).detach().cpu()
            xs.append(feats)
            if y is not None:
                ys.append(torch.as_tensor(y).long().cpu())
        X = torch.cat(xs, dim=0) if xs else torch.empty(0)
        Y = torch.cat(ys, dim=0) if ys else None
        return X, Y

    def _l2_normalize_rows(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if t.numel() == 0:
            return t
        return t / (t.norm(p=2, dim=1, keepdim=True) + eps)

    # ---- REF: class-mean prototypes (normalized) ----
    ref_X, ref_Y = _extract_feats_and_labels(ref_loader)
    if ref_X.numel() == 0 or ref_Y is None or ref_Y.numel() == 0:
        return float("nan")

    sums, counts = {}, {}
    for f, lab in zip(ref_X, ref_Y):
        k = int(lab)
        if k not in sums:
            sums[k] = f.clone()
            counts[k] = 1
        else:
            sums[k] += f
            counts[k] += 1
    if not sums:
        return float("nan")

    ref_keys  = sorted(sums.keys())
    ref_means = torch.stack([sums[k] / counts[k] for k in ref_keys], dim=0)  # [C, D]
    ref_means = _l2_normalize_rows(ref_means)  # prototypes (unit-norm)

    # ---- TARGET: unlabeled features (normalized) ----
    tgt_X, _ = _extract_feats_and_labels(target_loader)
    if tgt_X.numel() == 0:
        return float("nan")
    tgt_X = _l2_normalize_rows(tgt_X)  # unit-norm targets

    # ---- Assign each target to nearest prototype on normalized space ----
    dots = tgt_X @ ref_means.T                 # [N, C]
    assigned = dots.argmax(dim=1)              # [N]
    nearest_proto = ref_means[assigned]        # [N, D]

    # ---- Per-target L2 distance on normalized space; then mean ----
    per_target_dists = torch.norm(tgt_X - nearest_proto, p=2, dim=1)  # [N]
    return float(per_target_dists.mean().item())


# -----------------------------------------------------------------------------
# Core runner per dataset (restricted to selected T's)
# -----------------------------------------------------------------------------
def run_for_dataset(dataset: str, rng: random.Random, selected_Ts: Set[int]) -> Dict[str, object]:
    """
    Evaluate only at T in selected_Ts. For all other T, skip entirely.
    Returns:
        {
          "per_ckp": pd.DataFrame with columns ["ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"]
        }
    """
    df = load_matrix_csv(dataset)
    available_Ts = sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("ckp_"))
    chosen_Ts = sorted(t for t in selected_Ts if t in available_Ts)
    if not chosen_Ts:
        return {"per_ckp": pd.DataFrame(columns=["ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"])}

    methods = ["random","always_yes","always_no","ood_msp","class_mean","pick_higher"]
    rows = []

    # dataset-specific MSP threshold from ZS@ckp_1 (fallback to global)
    try:
        model_1, eval_loader_1, _ = build_datasets_and_model(dataset, 1)
        tau_msp_local = avg_msp(model_1, eval_loader_1)
    except Exception as e:
        print(f"[{dataset}] Failed to compute dataset MSP threshold from ZS@ckp_1: {e}")
        tau_msp_local = GLOBAL_TAU_MSP_DEFAULT
    print(f"[{dataset}] Using dataset-specific MSP threshold τ_msp={tau_msp_local:.4f}")

    acc_random, acc_yes, acc_no, acc_ood, acc_cls, acc_pick = [], [], [], [], [], []

    for T in trange(min(chosen_Ts), max(chosen_Ts)+1, desc=f"{_d2u(dataset)} | selective eval", leave=False):
        if T not in chosen_Ts:
            continue  # skip non-selected T entirely

        prev_k = max(1, T - 1)
        this_k = T

        row_out = {"ckp": f"ckp_{T}"}

        # A1 Random
        choose_yes = (rng.random() < 0.5)
        a = _acc_from_matrix(df, model_k=(this_k if choose_yes else prev_k), T=T)
        row_out["random"] = a
        if not math.isnan(a): acc_random.append(a)

        # A2 Always YES
        a = _acc_from_matrix(df, model_k=this_k, T=T)
        row_out["always_yes"] = a
        if not math.isnan(a): acc_yes.append(a)

        # A3 Always NO
        a = _acc_from_matrix(df, model_k=prev_k, T=T)
        row_out["always_no"] = a
        if not math.isnan(a): acc_no.append(a)

        # A4 MSP decision at T-1
        try:
            model_prev, eval_loader_prev, _ = build_datasets_and_model(dataset, prev_k)
            msp = avg_msp(model_prev, eval_loader_prev)
            decision_yes = (msp < tau_msp_local)
            a = _acc_from_matrix(df, model_k=(this_k if decision_yes else prev_k), T=T)
        except Exception as e:
            print(f"[{dataset}] Algo4 MSP failure at decision T-1={prev_k}: {e}")
            a = float("nan")
        row_out["ood_msp"] = a
        if not math.isnan(a): acc_ood.append(a)

        # A5 Class-mean (only if T>=3; else NaN)
        if T >= 3:
            try:
                model_prev, eval_loader_prev, train_pool_upto_prev_minus1 = build_datasets_and_model(dataset, prev_k)
                feat_model = FeatureHead(model_prev)
                dist = class_mean_distance_ref_vs_target(
                    feat_model,
                    ref_loader=train_pool_upto_prev_minus1,
                    target_loader=eval_loader_prev
                )
                decision_yes = (dist > TAU_FEAT)
                a = _acc_from_matrix(df, model_k=(this_k if decision_yes else prev_k), T=T)
            except Exception as e:
                print(f"[{dataset}] Algo5 class-mean failure at decision T-1={prev_k}: {e}")
                a = float("nan")
        else:
            a = float("nan")
        row_out["class_mean"] = a
        if not math.isnan(a): acc_cls.append(a)

        # A6 Pick-Higher oracle
        a_yes = _acc_from_matrix(df, model_k=this_k, T=T)
        a_no  = _acc_from_matrix(df, model_k=prev_k, T=T)
        if math.isnan(a_yes) and math.isnan(a_no):
            a_pick = float("nan")
        elif math.isnan(a_yes):
            a_pick = a_no
        elif math.isnan(a_no):
            a_pick = a_yes
        else:
            a_pick = max(a_yes, a_no)
        row_out["pick_higher"] = a_pick
        if not math.isnan(a_pick): acc_pick.append(a_pick)

        rows.append(row_out)

        # light cleanup
        del row_out, a_yes, a_no
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    df_ckp = pd.DataFrame(rows, columns=["ckp", *methods])
    return {"per_ckp": df_ckp}

# -----------------------------------------------------------------------------
# Build (dataset → set(T)) allow-list from CSV
# -----------------------------------------------------------------------------
def load_allowlist(csv_path: str) -> Dict[str, Set[int]]:
    """
    Returns: { dataset_underscore -> set({T}) }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Allow-list CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"dataset", "ckp"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Allow-list CSV must contain columns: {required_cols}")

    # Parse T from "ckp_XX"
    df = df.copy()
    df["T"] = df["ckp"].apply(_T_from_ckp)
    df = df[~df["T"].isna()]
    df["T"] = df["T"].astype(int)

    # Report stats
    num_pairs = len(df)
    num_datasets = df["dataset"].nunique()
    print(f"➡️  Using allow-list: {num_pairs} (dataset, ckp) pairs across {num_datasets} datasets.")

    # group
    mapping: Dict[str, Set[int]] = defaultdict(set)
    for _, r in df.iterrows():
        ds_u = str(r["dataset"])
        mapping[ds_u].add(int(r["T"]))
    return mapping

# -----------------------------------------------------------------------------
# Global per-ckp CSV (incremental)
# -----------------------------------------------------------------------------
SUMMARY_DIR   = "./when_to_adapt_out"
SUMMARY_PATH  = osp.join(SUMMARY_DIR, "summary_noTrain.csv")
SUMMARY_COLS  = ["dataset","ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"]

def _ensure_summary_header():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    if not osp.exists(SUMMARY_PATH):
        pd.DataFrame(columns=SUMMARY_COLS).to_csv(SUMMARY_PATH, index=False)

def _append_per_ckp_df(ds: str, df_ckp: pd.DataFrame):
    if df_ckp is None or df_ckp.empty:
        return
    df_out = df_ckp.copy()
    df_out.insert(0, "dataset", ds)
    # ensure correct column order/superset
    for col in SUMMARY_COLS:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out = df_out[SUMMARY_COLS]
    df_out.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)

# -----------------------------------------------------------------------------
# Batch driver (incremental I/O + aggressive cleanup)
# -----------------------------------------------------------------------------
def main():
    rng = random.Random(9527)
    os.makedirs("./when_to_adapt_out_filtered", exist_ok=True)
    os.makedirs(NP_RESULT_DIR, exist_ok=True)

    # OPTIONAL: start fresh each run
    # if osp.exists(SUMMARY_PATH):
    #     os.remove(SUMMARY_PATH)
    _ensure_summary_header()

    # Load (dataset_underscore -> set(T)) allow-list
    allow_map = load_allowlist(ALLOWLIST_CSV)

    used_pairs_counter = 0

    for ds in tqdm(DATASETS, desc="Datasets"):
        ds_u = _d2u(ds)
        selected_Ts = allow_map.get(ds_u, set())

        try:
            if not selected_Ts:
                print(f"[SKIP] {ds} has no selected checkpoints in allow-list.")
                # nothing to append globally if no selected checkpoints
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue

            out = run_for_dataset(ds, rng, selected_Ts)

            # write per-ckp CSV per dataset
            csv_path = os.path.join(NP_RESULT_DIR, f"{ds_u}_per_ckp.csv")
            out["per_ckp"].to_csv(csv_path, index=False)

            # append to GLOBAL per-ckp CSV (with dataset + ckp)
            _append_per_ckp_df(ds, out["per_ckp"])

            used_pairs_counter += len(out["per_ckp"])

        except Exception as e:
            print(f"[FAIL] {ds}: {e}")
            # If we know selected_Ts, append NaN rows for each T so gaps are visible
            if selected_Ts:
                fail_rows = []
                for T in sorted(selected_Ts):
                    fail_rows.append({
                        "ckp": f"ckp_{T}",
                        "random": float("nan"),
                        "always_yes": float("nan"),
                        "always_no": float("nan"),
                        "ood_msp": float("nan"),
                        "class_mean": float("nan"),
                        "pick_higher": float("nan"),
                    })
                df_fail = pd.DataFrame(fail_rows, columns=["ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"])
                _append_per_ckp_df(ds, df_fail)

        # ---- aggressive cleanup to avoid OOM on OSC ----
        for name in ["out", "csv_path", "selected_Ts"]:
            if name in locals():
                try:
                    del locals()[name]
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"✅ Incremental per-ckp results saved to {SUMMARY_PATH}")
    print(f"✅ Evaluated/appended ~{used_pairs_counter} (dataset, ckp) rows from the allow-list.")

if __name__ == "__main__":
    main()
