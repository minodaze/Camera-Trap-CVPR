# when_to_adapt_batch.py
# -*- coding: utf-8 -*-

"""
Evaluate 5 "when to adapt" algorithms over a list of datasets.

Algo 1  Random(YES/NO)
Algo 2  Always YES (train to get M(T+1))
Algo 3  Always NO  (reuse M(T))
Algo 4  OOD via avg Maximum Softmax Probability on D(T) under M(T)
        Decision: train if avg_MSP < TAU_MSP  (default 0.65)
Algo 5  Class-mean feature distance:
        distance between D(T) and pool{D(1), D(T-1)} (cosine)
        Decision: train if dist > TAU_FEAT (default 0.20)
        Starts at T >= 3

Each checkpoint decision is INDEPENDENT. Final metric per algorithm is the
average accuracy on d(T+1) across valid T for that dataset.

Assumptions / Conventions
- Accuracy matrix CSVs live at:
    {MATRIX_DIR}/{dataset_underscore}_matrix_balanced_accuracy.csv
  where dataset_underscore = dataset.replace("/", "_")
- Per-dataset YAML config lives at:
    {CONFIG_DIR}/{dataset_underscore}.yaml
- Trained models for accumulative runs are searched under:
    {yaml.log_path}/{dataset_underscore}/**/ckp_ckp_{T}_best_model.pth
  (If not found, we fall back to ZS weights; Algo 4/5 still run.)

Edit the constants below to match your filesystem.
"""

import os
import math
import glob
import json
import random
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from collections import defaultdict
from core import *
from run_pipeline import parse_args
from core import *
from core.module import get_al_module, get_cl_module, get_ood_module
from tqdm import tqdm, trange 


# ====== EDIT THESE PATHS ======================================================
MATRIX_DIR  = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only"
CONFIG_DIR  = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/finalConfig_np"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds (we'll sweep later)
TAU_MSP     = 0.65   # Algo 4: train if avg MSP < 0.65
TAU_FEAT    = 0.20   # Algo 5: train if class-mean cosine distance > 0.20

# Your dataset list
DATASETS = [
    "caltech/caltech_88",
    # "ENO/ENO_D06","serengeti/serengeti_N04","serengeti/serengeti_H11","APN/APN_K051",
    # "nz/nz_EFH_HCAME05","na/na_lebec_CA-21","MAD/MAD_A04","MTZ/MTZ_F04","APN/APN_13U",
    # "nz/nz_EFD_DCAMG03","nz/nz_PS1_CAM6213","nz/nz_EFH_HCAME09","MAD/MAD_D04",
    # "wellington/wellington_031c","KAR/KAR_A01","na/na_lebec_CA-18","serengeti/serengeti_Q09",
    # "APN/APN_U43B","APN/APN_K082","APN/APN_N1","nz/nz_EFH_HCAME08","serengeti/serengeti_L06",
    # "nz/nz_EFH_HCAMF01","caltech/caltech_46","serengeti/serengeti_S11","serengeti/serengeti_O13",
    # "PLN/PLN_B04","MTZ/MTZ_E05","nz/nz_EFD_DCAMH07","caltech/caltech_70","nz/nz_EFH_HCAMB05",
    # "KAR/KAR_B03","serengeti/serengeti_D02","MAD/MAD_B03","nz/nz_EFD_DCAMF06","caltech/caltech_88",
    # "na/na_lebec_CA-19","APN/APN_U23A","na/na_lebec_CA-05","nz/nz_EFH_HCAMI01","KGA/KGA_KHOGA04",
    # "ENO/ENO_C02","ENO/ENO_C04","MAD/MAD_C07","serengeti/serengeti_E05","serengeti/serengeti_V10",
    # "na/na_lebec_CA-31","serengeti/serengeti_F08","MAD/MAD_B06","nz/nz_EFH_HCAMB01",
    # "KGA/KGA_KHOLA03","nz/nz_EFH_HCAMD08","nz/nz_EFH_HCAMC03","serengeti/serengeti_L10",
    # "serengeti/serengeti_D09","idaho/idaho_122","serengeti/serengeti_Q11","MAD/MAD_H08",
    # "CDB/CDB_A05","serengeti/serengeti_E12","nz/nz_EFH_HCAMC02","serengeti/serengeti_T10",
    # "serengeti/serengeti_H03","nz/nz_PS1_CAM8008","na/na_lebec_CA-37","serengeti/serengeti_R10",
    # "nz/nz_EFD_DCAMH01","nz/nz_EFH_HCAMG13","serengeti/serengeti_K11","APN/APN_WM",
    # "nz/nz_PS1_CAM7312","ENO/ENO_E06","serengeti/serengeti_Q10","serengeti/serengeti_H08",
    # "APN/APN_TB17","serengeti/serengeti_Q07","caltech/caltech_38","MTZ/MTZ_D06",
    # "nz/nz_EFD_DCAMD10","MTZ/MTZ_D03",
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

def _model_row_name(T: int) -> str:
    return "zs" if T == 1 else f"model_{T}"

def _pick_accuracy(df: pd.DataFrame, model_T: int, test_Tp1: int) -> float:
    row = _model_row_name(model_T)
    col = _ckp_name(test_Tp1)
    if row not in df.index or col not in df.columns:
        return float("nan")
    v = df.loc[row, col]
    return float(v) if not pd.isna(v) else float("nan")

def load_matrix_csv(dataset: str) -> pd.DataFrame:
    csv_path = os.path.join(MATRIX_DIR, f"{_d2u(dataset)}_matrix_balanced_accuracy.csv")
    df = pd.read_csv(csv_path)
    # normalize
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

    # Ensure a couple of fields are set
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
        train_pool = train_ds.get_subset(is_train=True, ckp_list=_ckp_name(T - 1))
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
def avg_msp(model: nn.Module, loader) -> float:
    tot, n = 0.0, 0
    for batch in loader:
        # batch might be tuple/list or dict; expect first item is images, second labels
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
    """
    Returns raw features. We'll L2-normalize at the class-mean stage.
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        if hasattr(self.base, "forward_features"):
            feats = self.base.forward_features(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            feats = feats.float().view(feats.size(0), -1)
            return feats                      # <- raw
        else:
            logits = self.base(x)
            return logits.float()             # <- raw

@torch.no_grad()
def class_mean_distance_ref_vs_target(feat_model: nn.Module, ref_loader, target_loader) -> float:
    """
    L2 twice:
      (1) L2-normalize each per-class mean vector for REF and TARGET
      (2) Per-class distance = L2( mean_ref_norm - mean_tgt_norm ), then average
    """
    feat_model.eval()

    def class_means(loader):
        sums, counts = defaultdict(lambda: None), defaultdict(int)
        for batch in loader:
            if isinstance(batch, dict):
                x = batch.get("images") or batch.get("pixel_values") or batch.get("input")
                y = batch.get("labels") or batch.get("label")
            else:
                x, y = batch[0], batch[1]
            x = x.to(DEVICE, non_blocking=True)
            feats = feat_model(x)  # raw features [B, D]
            feats = feats.detach().cpu()
            for f, lab in zip(feats, y):
                lab = int(lab)
                if sums[lab] is None:
                    sums[lab] = f.clone()
                else:
                    sums[lab] += f
                counts[lab] += 1

        means = {}
        for k, c in counts.items():
            mu = sums[k] / c                         # class mean (raw)
            mu = mu / (mu.norm(p=2) + 1e-12)         # first L2: normalize mean
            means[k] = mu
        return means

    ref_mu    = class_means(ref_loader)      # pooled train D(1..T-1)
    target_mu = class_means(target_loader)   # eval@T

    common = sorted(set(ref_mu.keys()) & set(target_mu.keys()))
    if not common:
        return float("nan")

    dists = []
    for k in common:
        # second L2: Euclidean distance between normalized means
        d = (ref_mu[k] - target_mu[k]).norm(p=2).item()
        dists.append(d)

    return float(np.mean(dists))


# -----------------------------------------------------------------------------
# Core runner per dataset
# -----------------------------------------------------------------------------
def run_for_dataset(dataset: str, rng: random.Random) -> Dict[str, float]:
    df = load_matrix_csv(dataset)
    cols = [c for c in df.columns if c.startswith("ckp_")]
    if not cols:
        return {
            "summary": {k: float("nan") for k in ["random","always_yes","always_no","ood_msp","class_mean"]},
            "per_ckp": pd.DataFrame()
        }
    K = max(int(c.split("_")[1]) for c in cols)

    methods = ["random","always_yes","always_no","ood_msp","class_mean"]
    # per-ckp arrays (index 0..K-1 -> ckp_1..ckp_K)
    per_ckp = {m: [float("nan")] * K for m in methods}

    # --- include ZS@ckp_1 for ALL methods ---
    zs_ckp1 = df.loc["zs", "ckp_1"] if ("zs" in df.index and "ckp_1" in df.columns) else float("nan")
    if not math.isnan(zs_ckp1):
        for m in methods:
            per_ckp[m][0] = float(zs_ckp1)

    # --- include ZS@ckp_2 for Algo 5 ONLY (since A5 starts at T>=3) ---
    if K >= 2 and "ckp_2" in df.columns and "zs" in df.index:
        zs_ckp2 = float(df.loc["zs", "ckp_2"])
        if not math.isnan(zs_ckp2):
            per_ckp["class_mean"][1] = zs_ckp2

    acc_random, acc_yes, acc_no, acc_ood, acc_cls = [], [], [], [], []

    # Pre-fill accumulators with the ZS entries we just set
    if not math.isnan(per_ckp["random"][0]):     acc_random.append(per_ckp["random"][0])
    if not math.isnan(per_ckp["always_yes"][0]): acc_yes.append(per_ckp["always_yes"][0])
    if not math.isnan(per_ckp["always_no"][0]):  acc_no.append(per_ckp["always_no"][0])
    if not math.isnan(per_ckp["ood_msp"][0]):    acc_ood.append(per_ckp["ood_msp"][0])
    if not math.isnan(per_ckp["class_mean"][0]): acc_cls.append(per_ckp["class_mean"][0])
    if K >= 2 and not math.isnan(per_ckp["class_mean"][1]):  # the ZS@ckp_2 for A5
        acc_cls.append(per_ckp["class_mean"][1])

    # Algo 1–4: decisions at T=2..K-1, evaluated on ckp_{T+1} (index=T)
    for T in trange(2, K, desc=f"{_d2u(dataset)} | T-loop A1-4", leave=False):
        # Random
        a = _pick_accuracy(df, T + 1, T + 1) if (rng.random() < 0.5) else _pick_accuracy(df, T, T + 1)
        per_ckp["random"][T] = a
        if not math.isnan(a): acc_random.append(a)

        # Always YES
        a = _pick_accuracy(df, T + 1, T + 1)
        per_ckp["always_yes"][T] = a
        if not math.isnan(a): acc_yes.append(a)

        # Always NO
        a = _pick_accuracy(df, T, T + 1)
        per_ckp["always_no"][T] = a
        if not math.isnan(a): acc_no.append(a)

        # OOD MSP on eval@T -> decision -> accuracy on ckp_{T+1}
        try:
            model_T, eval_loader_T, _ = build_datasets_and_model(dataset, T)
            msp = avg_msp(model_T, eval_loader_T, desc=f"MSP T={T}")
            decision_yes = (msp < TAU_MSP)
            a = _pick_accuracy(df, T + 1 if decision_yes else T, T + 1)
        except Exception as e:
            print(f"[{dataset}] Algo4 failure at T={T}: {e}")
            a = float("nan")
        per_ckp["ood_msp"][T] = a
        if not math.isnan(a): acc_ood.append(a)

    # Algo 5: decisions at T=3..K-1, evaluated on ckp_{T+1} (index=T)
    for T in trange(3, K, desc=f"{_d2u(dataset)} | T-loop A5", leave=False):
        try:
            model_T, eval_loader_T, train_pool_loader = build_datasets_and_model(dataset, T)
            feat_model = FeatureHead(model_T)
            # ref=train pool (1..T-1); tgt=eval@T
            dist = class_mean_distance_ref_vs_target(
                feat_model, train_pool_loader, eval_loader_T,
                desc_ref=f"Means REF T<={T-1}", desc_tgt=f"Means TGT T={T}"
            )
            decision_yes = (dist > TAU_FEAT)
            a = _pick_accuracy(df, T + 1 if decision_yes else T, T + 1)
        except Exception as e:
            print(f"[{dataset}] Algo5 failure at T={T}: {e}")
            a = float("nan")
        per_ckp["class_mean"][T] = a
        if not math.isnan(a): acc_cls.append(a)

    def _mean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "random":     _mean(acc_random),
        "always_yes": _mean(acc_yes),
        "always_no":  _mean(acc_no),
        "ood_msp":    _mean(acc_ood),
        "class_mean": _mean(acc_cls),
    }

    # Build per-ckp DataFrame
    df_ckp = pd.DataFrame({
        "ckp": [f"ckp_{i+1}" for i in range(K)],
        **{m: per_ckp[m] for m in methods}
    })

    return {"summary": summary, "per_ckp": df_ckp}

# -----------------------------------------------------------------------------
# Batch driver
# -----------------------------------------------------------------------------
NP_RESULT_DIR = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result"

def main():
    rng = random.Random(42)
    os.makedirs("./when_to_adapt_out", exist_ok=True)
    os.makedirs(NP_RESULT_DIR, exist_ok=True)

    per_dataset_summary = {}
    for ds in tqdm(DATASETS, desc="Datasets"):
        try:
            out = run_for_dataset(ds, rng)
            per_dataset_summary[ds] = out["summary"]

            # --- NEW: write per-ckp CSV per dataset ---
            csv_path = os.path.join(NP_RESULT_DIR, f"{_d2u(ds)}_per_ckp.csv")
            out["per_ckp"].to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[FAIL] {ds}: {e}")
            per_dataset_summary[ds] = {k: float("nan") for k in ["random","always_yes","always_no","ood_msp","class_mean"]}

    # Global means
    keys = ["random","always_yes","always_no","ood_msp","class_mean"]
    global_mean = {}
    for k in keys:
        vals = [v[k] for v in per_dataset_summary.values() if not math.isnan(v[k])]
        global_mean[k] = float(np.mean(vals)) if vals else float("nan")

    # Print + save JSON summary
    print("\nPer-dataset means:")
    for ds, res in per_dataset_summary.items():
        print(ds, {k: round(v, 4) if not math.isnan(v) else None for k, v in res.items()})

    print("\nGlobal mean:")
    print({k: round(v, 4) if not math.isnan(v) else None for k, v in global_mean.items()})

    out = {"per_dataset": per_dataset_summary, "global_mean": global_mean,
           "thresholds": {"tau_msp": TAU_MSP, "tau_feat": TAU_FEAT}}
    with open("./when_to_adapt_out/summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print('Saved to ./when_to_adapt_out/summary.json')


if __name__ == "__main__":
    main()
