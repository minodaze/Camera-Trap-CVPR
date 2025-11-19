#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run "when to adapt" over ALL checkpoints that exist for each dataset.
- Accept datasets via CLI (--datasets CSV or --datasets_file FILE).
- Evaluate 5 policies (random, always_yes, always_no, ood_msp, class_mean, pick_higher).
- At each decision step (T uses info @ T-1), SAVE:
    * MSP vector (msp_scores.npy, float16) + msp_meta.json
    * Feature pack for CPU resweeps:
        - ref_means.npy (C,D, float16)  [L2-normalized class prototypes from REF train @ T-1]
        - tgt_feats_norm.npy (N,D, float16)  [L2-normalized TARGET eval @ T-1]
        - per_target_dists.npy (N, float16)  [||tgt - nearest_proto||_2]
        - feat_meta.json
- Writes a per-dataset summary CSV at: {SUMMARY_ROOT}/{DATASET_UNDERSCORE}/summary.csv
- Also writes a per-dataset per-ckp CSV under NP_RESULT_DIR: {DATASET_UNDERSCORE}_per_ckp.csv
"""

import os, os.path as osp, math, glob, json, random, sys, gc, argparse
from types import SimpleNamespace
from typing import Dict, List, Optional

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
from core import *                          # build_classifier, CkpDataset, etc.
from run_pipeline import parse_args         # your YAML arg parser

# ====== PATHS / CONFIG ========================================================
MATRIX_DIR       = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only"
CONFIG_DIR       = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/finalConfig_np"
NP_RESULT_DIR    = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_feature_saving_new_threshold"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds (can be swept later)
GLOBAL_TAU_MSP_DEFAULT = 0.65
TAU_FEAT  = 0.4   # Algo 5: train if mean per-target distance > 0.20
EPS_MSP   = 1e-12   # tiny tolerance for MSP comparison
EPS_FEAT  = 1e-12   # (optional) tolerance for feature-distance comparison

# Default dataset list (slash-form). You can override via CLI.
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

# Diagnostics/summary output
SUMMARY_ROOT = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_feature_saving_new_threshold/when_to_adapt_out"  # per-dataset folders live here
DIAG_DIR     = osp.join(SUMMARY_ROOT, "diagnostics")  # per-(dataset,T) artifacts

SAVE_PNG = False  # quick PCA plots off by default (CPU cost + matplotlib)

# -----------------------------------------------------------------------------
# CLI override for datasets
# -----------------------------------------------------------------------------
def parse_cli_overrides():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default="",
                   help="Comma-separated list like 'APN/APN_13U,serengeti/serengeti_H11,...'")
    p.add_argument("--datasets_file", type=str, default="",
                   help="Path to a file with one dataset per line.")
    args_cli, _unknown = p.parse_known_args()
    ds = []
    if args_cli.datasets_file:
        with open(args_cli.datasets_file, "r") as f:
            ds = [ln.strip() for ln in f if ln.strip()]
    elif args_cli.datasets:
        ds = [x.strip() for x in args_cli.datasets.split(",") if x.strip()]
    return ds

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
yaml_loader = YAML(typ="safe")

def _ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def _d2u(dataset: str) -> str: return dataset.replace("/", "_")
def _ckp_name(T: int) -> str:  return f"ckp_{T}"

def _dataset_summary_path(ds_u: str) -> str:
    d = osp.join(SUMMARY_ROOT, ds_u)
    _ensure_dir(d)
    return osp.join(d, "summary.csv")

def _diag_dir(ds_u: str, T: int) -> str:
    d = os.path.join(DIAG_DIR, ds_u, f"T_{T:02d}")
    _ensure_dir(d); return d

def _row_name_for_model_k(k: int) -> str: return "zs" if k == 1 else f"model_{k}"

def _acc_from_matrix(df: pd.DataFrame, model_k: int, T: int) -> float:
    row = _row_name_for_model_k(model_k)
    col = _ckp_name(T)
    if row not in df.index or col not in df.columns: return float("nan")
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
    with open(train_json, "r") as f: d = json.load(f)
    with open(test_json, "r") as f: d2 = json.load(f)
    d.update(d2)
    names = []
    for _, arr in d.items():
        for v in arr:
            name = v[label_key]
            if name not in names: names.append(name)
    return names

def build_zs_model(dataset: str) -> nn.Module:
    args = load_config(dataset)
    cc   = args.common_config
    class_names = get_class_names_from_json(
        cc["train_data_config_path"],
        cc["eval_data_config_path"],
        label_key = "common" if getattr(args, "label_type", "common") in ["common","common_name"] else "scientific"
    )
    zs_clf = build_classifier(args, class_names, args.device).to(DEVICE).eval()
    return zs_clf

def build_datasets_and_model(dataset: str, T: int):
    """
    Build:
      - classifier = M(T) (if weights exist), else zero-shot backbone
      - D(T) train loader for Algo 4
      - D(1), D(T-1), D(T) train loaders for Algo 5
    """
    args = load_config(dataset)
    cc   = args.common_config

    # classes
    class_names = get_class_names_from_json(
        cc["train_data_config_path"],
        cc["eval_data_config_path"],
        label_key = "common" if getattr(args, "label_type", "common") in ["common", "common_name"] else "scientific"
    )

    # Build classifier (zs)
    clf = build_classifier(args, class_names, args.device).to(DEVICE).eval()

    # Try load M(T) weights if available
    # Expect path like {args.log_path}/{dataset_underscore}/**/ckp_ckp_T_best_model.pth
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
            # Handle expanded head case with shape mismatch protection
            current_state_dict = clf.state_dict()

            # Filter out head parameters that have shape mismatches
            filtered_state_dict = {}
            for key, value in state.items():
                if key in current_state_dict:
                    current_shape = current_state_dict[key].shape
                    saved_shape = value.shape
                    
                    if current_shape == saved_shape:
                        # Shapes match, safe to load
                        filtered_state_dict[key] = value
                    elif 'head' in key:
                        # Head parameter with shape mismatch - handle carefully
                        if key == 'head.weight':
                            # Copy only the overlapping classes
                            min_classes = min(current_shape[0], saved_shape[0])
                            current_state_dict[key][:min_classes] = value[:min_classes]
                            print(f"Copied {min_classes} class weights for head.weight")
                        elif key == 'head.bias':
                            # Copy only the overlapping classes
                            min_classes = min(current_shape[0], saved_shape[0])
                            current_state_dict[key][:min_classes] = value[:min_classes]
                            print(f"Copied {min_classes} class biases for head.bias")
                        else:
                            print(f"Skipping head parameter {key} due to shape mismatch: {saved_shape} vs {current_shape}")
                    else:
                        print(f"Skipping parameter {key} due to shape mismatch: {saved_shape} vs {current_shape}")
                else:
                    print(f"Unexpected key in checkpoint: {key}")
            missing_keys, unexpected_keys = clf.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                head_missing = [k for k in missing_keys if 'head' in k]
                other_missing = [k for k in missing_keys if 'head' not in k]
                
                if head_missing:
                    print(f"Expected missing head parameters for expanded classes: {len(head_missing)} keys", Colors.YELLOW)
                if other_missing:
                    print(f"Unexpected missing parameters: {other_missing}")
                
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {unexpected_keys}")

            print("Model loaded successfully with expanded head handling")
            clf.to(args.device)
            clf.eval()
            # silently ignore expanded-head mismatches
        except Exception as e:
            print(f"[WARN] Failed to load weights ({weight_path}): {e}. Using ZS weights.")

    # Build Ckp datasets
    train_ds = CkpDataset(cc["train_data_config_path"], class_names, label_type=getattr(args, "label_type", "common"))
    eval_ds  = CkpDataset(cc["eval_data_config_path"],  class_names, is_train=False, label_type=args.label_type)
    # EVAL at T (single-ckp)  -> MSP target for Algo 4 AND target means for Algo 5
    eval_T = eval_ds.get_subset(is_train=False, ckp_list=_ckp_name(T))

    # TRAIN pool D(1..T-1): with is_train=True + ckp_list=f"ckp_{T-1}" gives accumulated 1..T-1
    if T > 1:
        accum_ckps = [_ckp_name(t) for t in range(1, T)]
        train_pool = train_ds.get_subset(is_train=True, ckp_list=accum_ckps) 
        print(f"load train_pool length {len(train_pool)} for T={T}")
    else:
        train_pool = train_ds.get_subset(is_train=True, ckp_list=_ckp_name(1))  # fallback for T=1, not really used

    bs_eval = int(cc.get("eval_batch_size", 256))
    dl_kw   = dict(batch_size=bs_eval, shuffle=False, num_workers=4)

    eval_loader_T = DataLoader(eval_T, **dl_kw)
    train_pool_loader_1_to_Tm1 = DataLoader(train_pool, **dl_kw)

    return clf, eval_loader_T, train_pool_loader_1_to_Tm1

# -----------------------------------------------------------------------------
# Measurements for Algo 4 & 5
# -----------------------------------------------------------------------------
@torch.no_grad()
class FeatureHead(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
    def forward(self, x):
        if hasattr(self.base, "forward_features"):
            feats = self.base.forward_features(x)
            if isinstance(feats, (list, tuple)): feats = feats[0]
            return feats.float().view(feats.size(0), -1)
        else:
            logits = self.base(x)
            return logits.float()

@torch.no_grad()
def compute_msp_scores(model: nn.Module, loader) -> np.ndarray:
    model.eval()
    scores = []
    for batch in loader:
        x = batch.get("images") if isinstance(batch, dict) else batch[0]
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        msp = probs.max(dim=-1).values.detach().cpu().numpy()
        scores.append(msp)
    if not scores: return np.empty((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32)

@torch.no_grad()
def extract_features(feat_model: nn.Module, loader, want_labels: bool=True):
    feat_model.eval()
    feats, labels = [], []
    for batch in loader:
        if isinstance(batch, dict):
            x = batch.get("images") or batch.get("pixel_values") or batch.get("input")
            y = batch.get("labels") if want_labels and ("labels" in batch or "label" in batch) else None
        else:
            x = batch[0]
            y = batch[1] if want_labels and len(batch) > 1 else None
        x = x.to(DEVICE, non_blocking=True)
        f = feat_model(x).detach().cpu().float()
        feats.append(f)
        if want_labels and y is not None:
            labels.append(torch.as_tensor(y).long().cpu())
    X = torch.cat(feats, dim=0) if feats else torch.empty(0)
    Y = torch.cat(labels, dim=0) if labels else None
    return X, Y

def _l2_normalize_rows(t: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    if t.numel() == 0: return t
    return t / (t.norm(p=2, dim=1, keepdim=True) + eps)

# -----------------------------------------------------------------------------
# Core runner per dataset (ALL ckps)
# -----------------------------------------------------------------------------
def run_for_dataset(dataset: str, rng: random.Random) -> Dict[str, object]:
    df = load_matrix_csv(dataset)
    available_Ts = sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("ckp_"))
    if not available_Ts:
        return {"per_ckp": pd.DataFrame(columns=["ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"])}

    zs_model = build_zs_model(dataset)
    zs_feat_model = FeatureHead(zs_model)

    K = max(available_Ts)
    methods = ["random","always_yes","always_no","ood_msp","class_mean","pick_higher"]
    rows = []

    rows.append({
        "ckp": "ckp_1",
        "random":      _acc_from_matrix(df, 1, 1),
        "always_yes":  _acc_from_matrix(df, 1, 1),
        "always_no":   _acc_from_matrix(df, 1, 1),
        "ood_msp":     _acc_from_matrix(df, 1, 1),
        "class_mean":  _acc_from_matrix(df, 1, 1),
        "pick_higher": _acc_from_matrix(df, 1, 1),
    })

    for T in trange(2, max(available_Ts)+1, desc=f"{_d2u(dataset)} | full eval", leave=False):
        if T not in available_Ts:
            continue  # skip holes

        prev_k = max(1, T - 1)
        this_k = T
        row_out = {"ckp": f"ckp_{T}"}

        # Build once per T-1 for both MSP & feature distance decisions
        model_prev, eval_loader_prev, train_pool_prev = build_datasets_and_model(dataset, prev_k)
        feat_model_prev = FeatureHead(model_prev)

        # ADD: per-ckp thresholds from the cached ZS model
        # τ_msp(T): ZS MSP mean on eval@(T-1)
        try:
            tau_msp_T = float(np.mean(compute_msp_scores(zs_model, eval_loader_prev)))
        except Exception as e:
            print(f"[{dataset}] Failed to compute ZS MSP threshold at T={T}: {e}")
            tau_msp_T = GLOBAL_TAU_MSP_DEFAULT

        # A1 Random
        choose_yes = (rng.random() < 0.5)
        a = _acc_from_matrix(df, model_k=(this_k if choose_yes else prev_k), T=T)
        row_out["random"] = a

        # A2 Always YES
        a = _acc_from_matrix(df, model_k=this_k, T=T)
        row_out["always_yes"] = a

        # A3 Always NO
        a = _acc_from_matrix(df, model_k=prev_k, T=T)
        row_out["always_no"] = a

        # A4 MSP decision + SAVE MSP vector (float16)
        try:
            ds_u = _d2u(dataset)
            out_dir = _diag_dir(ds_u, T)

            msp_scores = compute_msp_scores(model_prev, eval_loader_prev).astype(np.float16)
            np.save(os.path.join(out_dir, "msp_scores.npy"), msp_scores)

            msp_mean = float(np.mean(msp_scores)) if msp_scores.size else float("nan")
            if T == 2:
                decision_yes_msp = True  # always adapt at T=2
            else:
                decision_yes_msp = (not math.isnan(msp_mean)) and (msp_mean < (tau_msp_T + EPS_MSP))
            print(f"[{dataset}] T={T}: MSP mean={msp_mean:.4f} vs τ_msp={tau_msp_T:.4f} -> decision_yes={decision_yes_msp}")

            with open(os.path.join(out_dir, "msp_meta.json"), "w") as f:
                json.dump({
                    "dataset": ds_u,
                    "T": T,
                    "ckp_decision": f"ckp_{T-1}",
                    "tau_msp_in_use": float(tau_msp_T),
                    "n": int(msp_scores.size),
                    "stats": {
                        "mean": msp_mean,
                        "std": float(np.std(msp_scores)) if msp_scores.size else float("nan"),
                        "min": float(np.min(msp_scores)) if msp_scores.size else float("nan"),
                        "max": float(np.max(msp_scores)) if msp_scores.size else float("nan")
                    },
                    "decision_yes": bool(decision_yes_msp)
                }, f, indent=2)

            a = _acc_from_matrix(df, model_k=(this_k if decision_yes_msp else prev_k), T=T)
        except Exception as e:
            print(f"[{dataset}] Algo4 MSP failure at T={T}: {e}")
            a = float("nan")
        row_out["ood_msp"] = a

        # A5 Class-mean distance + SAVE feature pack for CPU resweeps
        if T >= 3:
            # τ_feat(T): ZS mean per-target distance using train 1..(T-1) prototypes vs eval@(T-1)
            try:
                zs_ref_X, zs_ref_Y = extract_features(zs_feat_model, train_pool_prev, want_labels=True)
                zs_tgt_X, _        = extract_features(zs_feat_model, eval_loader_prev,  want_labels=False)

                zs_ref_Xn = _l2_normalize_rows(zs_ref_X).numpy().astype(np.float32)
                zs_tgt_Xn = _l2_normalize_rows(zs_tgt_X).numpy().astype(np.float32)

                if zs_ref_Xn.size and zs_ref_Y is not None and zs_ref_Y.numel():
                    proto_list = []
                    for kcls in torch.unique(zs_ref_Y).tolist():
                        mu = zs_ref_X[zs_ref_Y == kcls].mean(dim=0, keepdim=False)
                        proto_list.append(mu)
                    zs_ref_means = torch.stack(proto_list, dim=0)
                    zs_ref_means = _l2_normalize_rows(zs_ref_means).numpy().astype(np.float32)
                else:
                    zs_ref_means = np.empty((0, zs_ref_Xn.shape[1] if zs_ref_Xn.size else 0), dtype=np.float32)

                if zs_tgt_Xn.size and zs_ref_means.size:
                    dots = zs_tgt_Xn @ zs_ref_means.T
                    assigned = np.argmax(dots, axis=1)
                    nearest  = zs_ref_means[assigned]
                    zs_per_d = np.linalg.norm(zs_tgt_Xn - nearest, ord=2, axis=1).astype(np.float32)
                    tau_feat_T = float(zs_per_d.mean()) if zs_per_d.size else float("nan")
                else:
                    tau_feat_T = float("nan")
            except Exception as e:
                print(f"[{dataset}] Failed to compute ZS feature threshold at T={T}: {e}")
                tau_feat_T = float("nan")

            # --- SAVE: ZS feature pack for Algo 5 (per-ckp) ---
            try:
                out_dir = _diag_dir(_d2u(dataset), T)
                # Only save if we actually computed something
                if 'zs_ref_means' in locals() and 'zs_tgt_Xn' in locals():
                    np.save(os.path.join(out_dir, "zs_ref_means.npy"),        (zs_ref_means if isinstance(zs_ref_means, np.ndarray) else np.array([])).astype(np.float16))
                    np.save(os.path.join(out_dir, "zs_tgt_feats_norm.npy"),   (zs_tgt_Xn   if isinstance(zs_tgt_Xn,   np.ndarray) else np.array([])).astype(np.float16))
                    # zs_per_d exists only when both sides are non-empty
                    if 'zs_per_d' in locals() and isinstance(zs_per_d, np.ndarray):
                        np.save(os.path.join(out_dir, "zs_per_target_dists.npy"), zs_per_d.astype(np.float16))
                    else:
                        np.save(os.path.join(out_dir, "zs_per_target_dists.npy"), np.empty((0,), dtype=np.float16))

                    with open(os.path.join(out_dir, "zs_feat_meta.json"), "w") as f:
                        json.dump({
                            "dataset": _d2u(dataset),
                            "T": T,
                            "ckp_decision": f"ckp_{T-1}",
                            "D": int(zs_ref_means.shape[1] if isinstance(zs_ref_means, np.ndarray) and zs_ref_means.size else (zs_tgt_Xn.shape[1] if isinstance(zs_tgt_Xn, np.ndarray) and zs_tgt_Xn.size else 0)),
                            "C": int(zs_ref_means.shape[0]) if isinstance(zs_ref_means, np.ndarray) and zs_ref_means.size else 0,
                            "N": int(zs_tgt_Xn.shape[0]) if isinstance(zs_tgt_Xn, np.ndarray) and zs_tgt_Xn.size else 0,
                            "dist_mean_zs": float(tau_feat_T),  # ZS-derived per-ckp threshold value
                        }, f, indent=2)
            except Exception as e:
                print(f"[{dataset}] Failed to save ZS feature pack at T={T}: {e}")


            try:
                ref_X, ref_Y = extract_features(feat_model_prev, train_pool_prev, want_labels=True)
                tgt_X, _     = extract_features(feat_model_prev, eval_loader_prev,  want_labels=False)

                ref_Xn = _l2_normalize_rows(ref_X).numpy().astype(np.float32)
                tgt_Xn = _l2_normalize_rows(tgt_X).numpy().astype(np.float32)

                if ref_Xn.size and ref_Y is not None and ref_Y.numel():
                    proto_list = []
                    for k in torch.unique(ref_Y).tolist():
                        mu = ref_X[ref_Y == k].mean(dim=0, keepdim=False)
                        proto_list.append(mu)
                    ref_means = torch.stack(proto_list, dim=0)
                    ref_means = _l2_normalize_rows(ref_means).numpy().astype(np.float32)
                else:
                    ref_means = np.empty((0, ref_Xn.shape[1] if ref_Xn.size else 0), dtype=np.float32)

                if tgt_Xn.size and ref_means.size:
                    dots = tgt_Xn @ ref_means.T
                    assigned = np.argmax(dots, axis=1)
                    nearest  = ref_means[assigned]
                    per_d    = np.linalg.norm(tgt_Xn - nearest, ord=2, axis=1).astype(np.float32)
                    dist_mean = float(per_d.mean()) if per_d.size else float("nan")
                else:
                    per_d = np.empty((0,), dtype=np.float32)
                    dist_mean = float("nan")

                out_dir = _diag_dir(_d2u(dataset), T)
                np.save(os.path.join(out_dir, "ref_means.npy"),        ref_means.astype(np.float16))
                np.save(os.path.join(out_dir, "tgt_feats_norm.npy"),   tgt_Xn.astype(np.float16))
                np.save(os.path.join(out_dir, "per_target_dists.npy"), per_d.astype(np.float16))

                with open(os.path.join(out_dir, "feat_meta.json"), "w") as f:
                    json.dump({
                        "dataset": _d2u(dataset),
                        "T": T,
                        "ckp_decision": f"ckp_{T-1}",
                        "D": int(ref_means.shape[1] if ref_means.size else (tgt_Xn.shape[1] if tgt_Xn.size else 0)),
                        "C": int(ref_means.shape[0]) if ref_means.size else 0,
                        "N": int(tgt_Xn.shape[0]) if tgt_Xn.size else 0,
                        "dist_mean_used": dist_mean,
                        "TAU_FEAT_used": float(tau_feat_T)
                    }, f, indent=2)

                decision_yes_feat = (not math.isnan(dist_mean)) and (dist_mean > (tau_feat_T + EPS_FEAT))
                print(f"[{dataset}] T={T}: feat dist mean={dist_mean:.4f} vs τ_feat={tau_feat_T:.4f} -> decision_yes={decision_yes_feat}")
                a = _acc_from_matrix(df, model_k=(this_k if decision_yes_feat else prev_k), T=T)
            except Exception as e:
                print(f"[{dataset}] Algo5 feature/logging failure at T={T}: {e}")
                a = float("nan")
            row_out["class_mean"] = a
        else:
            row_out["class_mean"] = _acc_from_matrix(df, model_k=2, T=2)

        # A6 Pick-Higher oracle
        a_yes = _acc_from_matrix(df, model_k=this_k, T=T)
        a_no  = _acc_from_matrix(df, model_k=prev_k, T=T)
        if math.isnan(a_yes) and math.isnan(a_no): a_pick = float("nan")
        elif math.isnan(a_yes): a_pick = a_no
        elif math.isnan(a_no):  a_pick = a_yes
        else:                   a_pick = max(a_yes, a_no)
        row_out["pick_higher"] = a_pick

        rows.append(row_out)

        # cleanup
        del row_out
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    df_ckp = pd.DataFrame(rows, columns=["ckp", *methods])
    return {"per_ckp": df_ckp}

# -----------------------------------------------------------------------------
# Per-dataset summary writer
# -----------------------------------------------------------------------------
SUMMARY_COLS = ["ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"]

def write_dataset_summary(ds: str, df_ckp: pd.DataFrame):
    """
    Writes a summary.csv inside {SUMMARY_ROOT}/{ds_u}/summary.csv with columns:
    ["ckp","random","always_yes","always_no","ood_msp","class_mean","pick_higher"]
    """
    ds_u = _d2u(ds)
    out_dir = osp.join(SUMMARY_ROOT, ds_u)
    _ensure_dir(out_dir)
    path = osp.join(out_dir, "summary.csv")
    # ensure column order/superset
    for col in SUMMARY_COLS:
        if col not in df_ckp.columns:
            df_ckp[col] = np.nan
    df_ckp = df_ckp[SUMMARY_COLS]
    df_ckp.to_csv(path, index=False)
    print(f"💾 Wrote per-dataset summary: {path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    rng = random.Random(9527)
    _ensure_dir(SUMMARY_ROOT)
    _ensure_dir(DIAG_DIR)
    _ensure_dir(NP_RESULT_DIR)

    ds_override = parse_cli_overrides()
    datasets_to_run = ds_override if ds_override else DATASETS

    total_rows = 0
    for ds in tqdm(datasets_to_run, desc="Datasets"):
        ds_u = _d2u(ds)
        try:
            out = run_for_dataset(ds, rng)
            # 1) per-dataset per-ckp CSV (flat file)
            csv_path = os.path.join(NP_RESULT_DIR, f"{ds_u}_per_ckp.csv")
            out["per_ckp"].to_csv(csv_path, index=False)
            # 2) per-dataset summary in its own folder
            write_dataset_summary(ds, out["per_ckp"])
            total_rows += len(out["per_ckp"])
        except Exception as e:
            print(f"[FAIL] {ds}: {e}")

        # cleanup
        for name in ["out", "csv_path"]:
            if name in locals():
                try: del locals()[name]
                except Exception: pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"✅ Finished. Wrote per-dataset summaries under {SUMMARY_ROOT}")
    print(f"✅ Total rows across datasets: ~{total_rows}")

if __name__ == "__main__":
    main()
