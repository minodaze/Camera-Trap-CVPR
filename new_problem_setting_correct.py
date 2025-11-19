# when_to_adapt_batch.py
# -*- coding: utf-8 -*-

"""
Evaluate 5 "when to adapt" algorithms over a list of datasets.

DECISION POLICY (final):
- We decide for checkpoint T using ONLY info up to T-1, then we evaluate on ckp_T.
- Matrix rows:
    zs            -> zero-shot (no training)
    model_k       -> model trained on D(1..k-1)
- Matrix columns:
    ckp_T         -> test set from checkpoint T

Algorithms
----------
Algo 1  Random (YES/NO) at T-1
        YES -> use model_T on ckp_T
        NO  -> use model_{T-1} on ckp_T

Algo 2  Always YES at T-1
        -> use model_T on ckp_T

Algo 3  Always NO at T-1
        -> use model_{T-1} on ckp_T

Algo 4  OOD via average Maximum Softmax Probability (MSP)
        Decision made at T-1, using M(T-1) on eval@(T-1).
        If avg_MSP < TAU_MSP -> YES (train) -> model_T on ckp_T
        else                 -> NO          -> model_{T-1} on ckp_T

Algo 5  Class-mean feature distance (L2 twice)
        Decision made at T-1, using M(T-1).
        REF    = pooled train D(1..T-2)
        TARGET = eval@(T-1)
        Distance = mean over classes of L2( L2norm(mean_REF) - L2norm(mean_TARGET) )
        If dist > TAU_FEAT -> YES (train) -> model_T on ckp_T
        else               -> NO          -> model_{T-1} on ckp_T
        Special cases:
          - ckp_1 = ZS
          - ckp_2 = model_2@ckp_2  (we can't form REF D(1..0), so default to train)

Outputs
-------
1) A JSON with per-dataset method means and a global mean at ./when_to_adapt_out/summary.json
2) A per-ckp CSV per dataset at:
   /fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result/<dataset_underscore>_per_ckp.csv

Assumptions / Conventions
-------------------------
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
from typing import Dict, List
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
MATRIX_DIR   = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only"
CONFIG_DIR   = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/finalConfig_np"
NP_RESULT_DIR= "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_basic"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds (can be swept later)
LOBAL_TAU_MSP_DEFAULT = 0.65
TAU_FEAT  = 0.20   # Algo 5: train if class-mean distance > 0.20

# Your dataset list
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
    "CDB/CDB_A05","serengeti/serengeti_E12","nz/nz_EFH_HCAMC02","serengeti/serengeti_T10",
    "serengeti/serengeti_H03","nz/nz_PS1_CAM8008","na/na_lebec_CA-37","serengeti/serengeti_R10",
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

def _row_name_for_model_k(k: int) -> str:
    return "zs" if k == 1 else f"model_{k}"

def _acc_from_matrix(df: pd.DataFrame, model_k: int, T: int) -> float:
    """Read accuracy matrix row=model_k, col=ckp_T."""
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
    """Use your repo's parse_args to load a YAML into an args namespace consistently."""
    cfg_path = os.path.join(CONFIG_DIR, f"{_d2u(dataset)}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    argv_backup = sys.argv[:]
    try:
        sys.argv = [argv_backup[0], "--c", cfg_path]
        args = parse_args()
    finally:
        sys.argv = argv_backup

    # Ensure sane defaults
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
    """Return raw (pre-softmax) features from the base classifier."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        if hasattr(self.base, "forward_features"):
            print("Using forward_features to extract features.")
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
            feats = feat_model(x).detach().cpu()
            for f, lab in zip(feats, y):
                lab = int(lab)
                if sums[lab] is None:
                    sums[lab] = f.clone()
                else:
                    sums[lab] += f
                counts[lab] += 1

        means = {}
        for k, c in counts.items():
            mu = sums[k] / c
            mu = mu / (mu.norm(p=2) + 1e-12)  # first L2-normalization
            means[k] = mu
        return means

    ref_mu    = class_means(ref_loader)
    tgt_mu    = class_means(target_loader)

    common = sorted(set(ref_mu.keys()) & set(tgt_mu.keys()))
    if not common:
        return float("nan")

    dists = []
    for k in common:
        d = (ref_mu[k] - tgt_mu[k]).norm(p=2).item()  # second L2 (distance between normalized means)
        dists.append(d)

    return float(np.mean(dists))


# -----------------------------------------------------------------------------
# Core runner per dataset
# -----------------------------------------------------------------------------
def run_for_dataset(dataset: str, rng: random.Random) -> Dict[str, object]:
    df = load_matrix_csv(dataset)
    ckp_cols = [c for c in df.columns if c.startswith("ckp_")]
    if not ckp_cols:
        return {"summary": {k: float("nan") for k in ["random","always_yes","always_no","ood_msp","class_mean","pick_higher"]},
                "per_ckp": pd.DataFrame()}

    K = max(int(c.split("_")[1]) for c in ckp_cols)

    methods = ["random","always_yes","always_no","ood_msp","class_mean","pick_higher"]
    per_ckp = {m: [float("nan")] * K for m in methods}

    # --- ckp_1 = ZS for all methods
    per_ckp_val_ckp1 = _acc_from_matrix(df, model_k=1, T=1)
    for m in methods:
        per_ckp[m][0] = per_ckp_val_ckp1

    # --- Algo 5 special case: ckp_2 = model_2@ckp_2 (default-to-train)
    if K >= 2:
        per_ckp["class_mean"][1] = _acc_from_matrix(df, model_k=2, T=2)

    # === NEW: dataset-specific MSP threshold τ_msp = avg_msp(ZS, eval@ckp_1)
    try:
        model_1, eval_loader_1, _ = build_datasets_and_model(dataset, 1)  # M(1)=ZS, eval@1
        tau_msp_local = avg_msp(model_1, eval_loader_1)
    except Exception as e:
        print(f"[{dataset}] Failed to compute dataset MSP threshold from ZS@ckp_1: {e}")
        tau_msp_local = LOBAL_TAU_MSP_DEFAULT # fallback to global
    print(f"[{dataset}] Using dataset-specific MSP threshold τ_msp={tau_msp_local:.4f}")

    acc_random, acc_yes, acc_no, acc_ood, acc_cls, acc_pick = [], [], [], [], [], []
    # seed accumulators
    if not math.isnan(per_ckp["random"][0]):     acc_random.append(per_ckp["random"][0])
    if not math.isnan(per_ckp["always_yes"][0]): acc_yes.append(per_ckp["always_yes"][0])
    if not math.isnan(per_ckp["always_no"][0]):  acc_no.append(per_ckp["always_no"][0])
    if not math.isnan(per_ckp["ood_msp"][0]):    acc_ood.append(per_ckp["ood_msp"][0])
    if not math.isnan(per_ckp["class_mean"][0]): acc_cls.append(per_ckp["class_mean"][0])
    if not math.isnan(per_ckp["pick_higher"][0]): acc_pick.append(per_ckp["pick_higher"][0])
    if K >= 2 and not math.isnan(per_ckp["class_mean"][1]):
        acc_cls.append(per_ckp["class_mean"][1])

    # ----- Fill ckp_T for ALL methods (T >= 2) -----
    for T in trange(2, K+1, desc=f"{_d2u(dataset)} | decisions T-1 -> eval T", leave=False):
        prev_k = T - 1  # NO-path
        this_k = T      # YES-path

        # A1 Random
        choose_yes = (rng.random() < 0.5)
        a = _acc_from_matrix(df, model_k=(this_k if choose_yes else prev_k), T=T)
        per_ckp["random"][T-1] = a
        if not math.isnan(a): acc_random.append(a)

        # A2 Always YES
        a = _acc_from_matrix(df, model_k=this_k, T=T)
        per_ckp["always_yes"][T-1] = a
        if not math.isnan(a): acc_yes.append(a)

        # A3 Always NO
        a = _acc_from_matrix(df, model_k=prev_k, T=T)
        per_ckp["always_no"][T-1] = a
        if not math.isnan(a): acc_no.append(a)

        # A4 MSP: decide at T-1 using M(T-1) on eval@(T-1)
        try:
            model_prev, eval_loader_prev, train_pool_upto_prev_minus1 = build_datasets_and_model(dataset, prev_k)
            msp = avg_msp(model_prev, eval_loader_prev)
            print(f"[{dataset}] Algo4 MSP at decision T-1={prev_k}: avg_MSP={msp:.4f}")
            decision_yes = (msp < tau_msp_local)  # low MSP -> drift -> train
            # decision_yes = False  # DISABLE ALGO 4 DECISION LOGIC FOR TESTING
            a = _acc_from_matrix(df, model_k=(this_k if decision_yes else prev_k), T=T)
        except Exception as e:
            print(f"[{dataset}] Algo4 MSP failure at decision T-1={prev_k}: {e}")
            a = float("nan")
        per_ckp["ood_msp"][T-1] = a
        if not math.isnan(a): acc_ood.append(a)

        # A5 Class-mean (starts at T>=3): REF D(1..T-2) vs TARGET eval@(T-1), features by M(T-1)
        if T >= 3:
            try:
                model_prev, eval_loader_prev, train_pool_upto_prev_minus1 = build_datasets_and_model(dataset, prev_k)
                feat_model = FeatureHead(model_prev)
                dist = class_mean_distance_ref_vs_target(
                    feat_model,
                    ref_loader=train_pool_upto_prev_minus1,  # D(1..T-2)
                    target_loader=eval_loader_prev          # eval@(T-1)
                )
                print(f"[{dataset}] Algo5 class-mean at decision T-1={prev_k}: distance={dist:.4f}")
                decision_yes = (dist > TAU_FEAT)
                # decision_yes = False  # DISABLE ALGO 5 DECISION LOGIC FOR TESTING
                a = _acc_from_matrix(df, model_k=(this_k if decision_yes else prev_k), T=T)
            except Exception as e:
                print(f"[{dataset}] Algo5 class-mean failure at decision T-1={prev_k}: {e}")
                a = float("nan")
            per_ckp["class_mean"][T-1] = a
            if not math.isnan(a): acc_cls.append(a)
        # T==2 is already set: model_2@ckp_2

        # A6 Pick-Higher (oracle upper bound)
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
        per_ckp["pick_higher"][T-1] = a_pick
        if not math.isnan(a_pick): acc_pick.append(a_pick)

    def _mean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "random":     _mean(acc_random),
        "always_yes": _mean(acc_yes),
        "always_no":  _mean(acc_no),
        "ood_msp":    _mean(acc_ood),
        "class_mean": _mean(acc_cls),
        "pick_higher": _mean(acc_pick),
    }

    df_ckp = pd.DataFrame({
        "ckp": [f"ckp_{i+1}" for i in range(K)],
        **{m: per_ckp[m] for m in methods}
    })

    return {"summary": summary, "per_ckp": df_ckp}


# -----------------------------------------------------------------------------
# Batch driver
# -----------------------------------------------------------------------------
def main():
    rng = random.Random(9527)
    os.makedirs("./when_to_adapt_out", exist_ok=True)
    os.makedirs(NP_RESULT_DIR, exist_ok=True)

    per_dataset_summary = {}

    for ds in tqdm(DATASETS, desc="Datasets"):
        try:
            out = run_for_dataset(ds, rng)
            per_dataset_summary[ds] = out["summary"]

            # write per-ckp CSV per dataset
            csv_path = os.path.join(NP_RESULT_DIR, f"{_d2u(ds)}_per_ckp.csv")
            out["per_ckp"].to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[FAIL] {ds}: {e}")
            per_dataset_summary[ds] = {k: float("nan") for k in ["random","always_yes","always_no","ood_msp","class_mean","pick_higher"]}

# build summary.csv (one row per dataset)
    df_summary = pd.DataFrame.from_dict(per_dataset_summary, orient="index")
    df_summary.index.name = "dataset"
    df_summary.reset_index(inplace=True)
    df_summary.to_csv("./when_to_adapt_out/summary.csv", index=False)
    print('Saved per-dataset means to ./when_to_adapt_out/summary.csv')

    # (optional) print to console
    print("\nPer-dataset means:")
    for _, row in df_summary.iterrows():
        ds = row["dataset"]
        print(ds, {k: (None if pd.isna(row[k]) else round(float(row[k]), 4))
                   for k in ["random","always_yes","always_no","ood_msp","class_mean","pick_higher"]})


if __name__ == "__main__":
    main()
