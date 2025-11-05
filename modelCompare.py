#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-dataset accuracy matrices using the SAME build pattern as before:
classifier = build_classifier(base_args, class_names, device=base_args.device, other_pretrain_weight=...)
- Rows:  ["zs", "ckp_2", ..., "ckp_K"]
- Cols:  ["ckp_1", ..., "ckp_K"]
- Value: accuracy from eval() (not balanced); future-use cells = NaN
- Per-dataset CSV: {args.log_root}/{GROUP_DATASET}/acc_matrix.csv
"""

import os
import re
import glob
import json
import numpy as np
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== YOUR DATASET LIST HERE ====
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


    "APN/APN_K023",
    "APN/APN_K051",
    "APN/APN_K082",
    "APN/APN_U64C",
    "APN/APN_WM",
    "caltech/caltech_38",
    "ENO/ENO_C05",
    "ENO/ENO_D06",
    "idaho/idaho_105",
    "idaho/idaho_122",
    "idaho/idaho_85",
    "KGA/KGA_KHOGA04",
    "KGA/KGA_KHOGB07",
    "KGA/KGA_KHOGC05",
    "KGA/KGA_KHOLA03",
    "KGA/KGA_KHOLA08",
    "MAD/MAD_A05",
    "nz/nz_EFD_DCAMC03",
    "nz/nz_EFD_DCAMC04"
]
# ================================

# ---- imports from your project (same as training) ----
from core import *  # build_classifier, CkpDataset, eval, print_metrics

def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Model Comparison Matrix Eval")

    # ---- paths / layout ----
    p.add_argument("--model-root", type=str,
                   default="/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_accum",
                   help="Root containing per-dataset checkpoint folders")
    p.add_argument("--model-tail", type=str,
                #    default="best_accum/bioclip2/lora_8_text_head/all/log",
                default="",
                   help="Tail path under {GROUP_DATASET} to reach checkpoint dir")
    p.add_argument("--data-root", type=str,
                   default="/fs/scratch/PAS2099/camera-trap-benchmark/dataset",
                   help="Root for dataset JSONs (train/test.json)")
    p.add_argument("--split-dir", type=str, default="30",
                   help="Subdir under each dataset where the JSONs live")
    p.add_argument("--log-root", type=str,
                   default="/fs/scratch/PAS2099/Lemeng/icicle/model_comparison_outputs",
                   help="Where to write per-dataset outputs")

    # ---- model/build flags commonly expected by your pipeline ----
    p.add_argument('--pretrained_weights', type=str, default='bioclip2',
                   choices=['bioclip', 'bioclip2'])
    p.add_argument('--full', action='store_true')

    # Interpolation knobs (compat)
    p.add_argument('--interpolation_model', action='store_false')
    p.add_argument('--interpolation_head', action='store_false')
    p.add_argument('--interpolation_alpha', type=float, default=0.5)

    # Misc
    p.add_argument('--seed', type=int, default=9527)
    p.add_argument('--gpu_memory_monitor', action='store_false')
    p.add_argument('--no_gpu_monitor_colors', action='store_false')
    p.add_argument('--label_type', type=str, default='common')

    # PEFT / adapters (kept for compatibility)
    p.add_argument('--lora_bottleneck', type=int, default=0)
    p.add_argument('--ft_attn_module', default=None, choices=['adapter','convpass','repadapter'])
    p.add_argument('--ft_attn_mode', default='parallel', choices=['parallel','sequential_after','sequential_before'])
    p.add_argument('--ft_attn_ln', default='before', choices=['before','after'])
    p.add_argument('--ft_mlp_module', default=None, choices=['adapter','convpass','repadapter'])
    p.add_argument('--ft_mlp_mode', default='parallel', choices=['parallel','sequential_after','sequential_before'])
    p.add_argument('--ft_mlp_ln', default='before', choices=['before','after'])
    p.add_argument('--adapter_bottleneck', type=int, default=64)
    p.add_argument('--adapter_init', type=str, default='lora_kaiming', choices=['lora_kaiming','xavier','zero','lora_xavier'])
    p.add_argument('--adapter_scaler', type=float, default=0.1)
    p.add_argument('--convpass_xavier_init', action='store_true')
    p.add_argument('--convpass_bottleneck', type=int, default=8)
    p.add_argument('--convpass_init', type=str, default='lora_xavier', choices=['lora_kaiming','xavier','zero','lora_xavier'])
    p.add_argument('--convpass_scaler', type=float, default=10.0)
    p.add_argument('--vpt_mode', type=str, default=None, choices=['deep','shallow'])
    p.add_argument('--vpt_num', type=int, default=10)
    p.add_argument('--vpt_layer', type=int, default=None)
    p.add_argument('--vpt_dropout', type=float, default=0.1)
    p.add_argument('--ssf', action='store_true')
    p.add_argument('--bitfit', action='store_true')
    p.add_argument('--difffit', action='store_true')
    p.add_argument('--fact_dim', type=int, default=8)
    p.add_argument('--fact_type', type=str, default=None, choices=['tk','tt'])
    p.add_argument('--fact_scaler', type=float, default=1.0)
    p.add_argument('--repadapter_bottleneck', type=int, default=8)
    p.add_argument('--repadapter_init', type=str, default='lora_xavier', choices=['lora_xavier','lora_kaiming','xavier','zero'])
    p.add_argument('--repadapter_scaler', type=float, default=1.0)
    p.add_argument('--repadapter_group', type=int, default=2)
    p.add_argument('--vqt_num', type=int, default=0)
    p.add_argument('--vqt_dropout', type=float, default=0.1)
    p.add_argument('--mlp_index', type=int, nargs='+', default=None)
    p.add_argument('--mlp_type', type=str, default='full', choices=['fc1','fc2','full'])
    p.add_argument('--attention_index', type=int, nargs='+', default=None)
    p.add_argument('--attention_type', type=str, default='full', choices=['qkv','proj','full'])
    p.add_argument('--ln', action='store_true')

    # ---- inference / runtime ----
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)

    # ---- optional: only one test checkpoint label ----
    p.add_argument("--only-ckp", type=str, default=None)

    p.add_argument('--text', type=str, default='head', choices=['head','full','lora'])
    p.add_argument('--text_template', type=str, default='openai', choices=['bioclip','openai'])
    p.add_argument('--model_dir', type=str, help='(unused here)')

    p.add_argument('--merge_factor', default=1, type=float)
    p.add_argument('--debug', action='store_true')

    return p.parse_args()

# -------- helpers --------

def group_dataset_to_names(key: str) -> Tuple[str, str, str]:
    if "/" not in key:
        raise ValueError(f"Bad dataset key '{key}', expected 'GROUP/DATASET'")
    g, d = key.split("/", 1)
    return g, d, f"{g}_{d}"

def dataset_paths(args, group: str, dataset: str, combo: str):
    model_dir = os.path.join(args.model_root, combo, args.model_tail)
    data_dir  = os.path.join(args.data_root, group, dataset, args.split_dir)
    train_json = os.path.join(data_dir, "train.json")
    test_json  = os.path.join(data_dir, "test.json")
    out_dir    = os.path.join(args.log_root, combo)
    os.makedirs(out_dir, exist_ok=True)
    return model_dir, train_json, test_json, out_dir

def discover_ckps(ckp_dir: str):
    ts = set()
    for fp in glob.glob(os.path.join(ckp_dir, "ckp_ckp_*_best_model.pth")):
        m = re.search(r"ckp_ckp_(\d+)_best_model\.pth$", os.path.basename(fp))
        if m: ts.add(int(m.group(1)))
    for fp in glob.glob(os.path.join(ckp_dir, "ckp_*_best_model.pth")):
        m = re.search(r"ckp_(\d+)_best_model\.pth$", os.path.basename(fp))
        if m: ts.add(int(m.group(1)))
    return sorted(ts)

def load_class_names(train_json: str, test_json: str, label_key: str):
    with open(train_json, "r") as f: t = json.load(f)
    with open(test_json, "r") as f: e = json.load(f)
    seen, classes = set(), []
    for dct in (t, e):
        for _, lst in dct.items():
            for x in lst:
                lab = x[label_key]
                if lab not in seen:
                    seen.add(lab); classes.append(lab)
    return classes

def eval_on_ckp_subset(classifier, dset, ckp_label, device, batch_size, num_workers, num_classes):
    try:
        sub = dset.get_subset(is_train=False, ckp_list=ckp_label)
    except Exception:
        sub = dset.get_subset(is_train=False, ckp_list=[ckp_label])
    if len(sub) == 0:
        return np.nan
    loader = DataLoader(
        sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(torch.device(device).type == "cuda"),
    )
    loss_arr, preds_arr, labels_arr = eval(classifier, loader, device, chop_head=False)
    acc, bal_acc = print_metrics(loss_arr, preds_arr, labels_arr, num_classes, log_predix=f"[{ckp_label}] ")
    return float(bal_acc)

# -------- main per-dataset --------

def run_one_dataset(base_args, key: str):
    group, dataset, combo = group_dataset_to_names(key)
    model_dir, train_json, test_json, out_dir = dataset_paths(base_args, group, dataset, combo)

    if not os.path.isdir(model_dir):
        print(f"[WARN] {combo}: model dir not found: {model_dir}; skip.")
        return
    if not (os.path.isfile(train_json) and os.path.isfile(test_json)):
        print(f"[WARN] {combo}: train/test JSON missing; skip.")
        return

    class_names = load_class_names(train_json, test_json, label_key="common")
    print(f"[{combo}] #classes={len(class_names)}")

    eval_dset = CkpDataset(test_json, class_names, is_crop=False, label_type="common")

    device = torch.device(base_args.device if (base_args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    t_list = discover_ckps(model_dir)
    if not t_list:
        print(f"[{combo}] No checkpoints found in {model_dir}")
        return
    K = max(t_list) if t_list else 1

    row_tags = ["zs"] + ([f"model_{t}" for t in range(2, K+1)] if K >= 2 else [])
    col_tags = [f"ckp_{t}" for t in range(1, K+1)]
    M = np.full((len(row_tags), len(col_tags)), np.nan, dtype=float)

    device = base_args.device if (base_args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # ---- ZS row ----
    base_args.lora_bottleneck = 0
    zs_clf = build_classifier(base_args, class_names, device=base_args.device, other_pretrain_weight=False).to(device)
    zs_clf.eval()
    for j, ckp_j in enumerate(tqdm(col_tags, desc=f"{combo} | ZS eval", leave=False)):
        if base_args.only_ckp is not None and ckp_j != base_args.only_ckp:
            continue
        M[0, j] = eval_on_ckp_subset(zs_clf, eval_dset, ckp_j, device, base_args.eval_batch_size, base_args.num_workers,len(class_names))

    # ---- trained rows: load ckp_t weights; do not use future columns ----
    ft_bottleneck = base_args.lora_bottleneck if base_args.lora_bottleneck > 0 else 8
    for i, tag in enumerate(tqdm(row_tags[1:], desc=f"{combo} | FT rows", leave=False), start=1):
        t = int(tag.split("_")[1])
        base_args.lora_bottleneck = ft_bottleneck
        clf = build_classifier(base_args, class_names, device=base_args.device, other_pretrain_weight=True).to(device)
        clf.eval()

        cand = [
            os.path.join(model_dir, f"ckp_ckp_{t}_best_model.pth"),
            os.path.join(model_dir, f"ckp_{t}_best_model.pth"),
        ]
        ckpt_path = next((p for p in cand if os.path.exists(p)), None)
        if ckpt_path is None:
            print(f"[WARN] {combo}: missing weights for {tag}; row left NaN.")
            continue
        try:
            clf.load_state_dict(torch.load(ckpt_path, map_location=device))
        except Exception as e:
            print(f"[ERROR] {combo}: failed to load {ckpt_path}: {e}")
            continue

        for j, ckp_j in enumerate(col_tags):
            tj = int(ckp_j.split("_")[1])
            if tj < t:
                continue
            if base_args.only_ckp is not None and ckp_j != base_args.only_ckp:
                continue
            M[i, j] = eval_on_ckp_subset(clf, eval_dset, ckp_j, device, base_args.eval_batch_size, base_args.num_workers,len(class_names))

    # ---- save CSV ----
    out_csv = os.path.join(out_dir, "acc_matrix.csv")
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model \\ test_ckp"] + col_tags)
        for i, rtag in enumerate(row_tags):
            row = [rtag] + [("" if np.isnan(v) else f"{v:.4f}") for v in M[i]]
            w.writerow(row)
    print(f"[OK] {combo}: saved matrix -> {out_csv}")

    # ---- simple diagonal check (|diff| > 0.02) after CSV is written ----
    mismatch_count = 0
    check_path = os.path.join(model_dir, "final_training_summary.json")
    if os.path.isfile(check_path):
        try:
            with open(check_path, "r") as f:
                summary = json.load(f)
            ckps_json = summary.get("checkpoint_results", {})
            for t in range(2, len(col_tags) + 1):
                rtag = f"model_{t}"
                ctag = f"ckp_{t}"
                if rtag in row_tags and ctag in col_tags:
                    i = row_tags.index(rtag)
                    j = col_tags.index(ctag)
                    if not np.isnan(M[i, j]):
                        exp = ckps_json.get(ctag, {}).get("balanced_accuracy")
                        if exp is not None and abs(float(M[i, j]) - float(exp)) > 0.02:
                            mismatch_count += 1
        except Exception as e:
            print(f"[WARN] {combo}: simple diagonal check failed: {e}")
    else:
        print(f"[WARN] {combo}: final_training_summary.json not found; skip simple diagonal check.")

    return combo, mismatch_count



def main():
    base_args = parse_args()
    results = []
    total = 0
    for key in tqdm(DATASETS, desc="Datasets"):
        try:
            combo, mismatches = run_one_dataset(base_args, key)
            results.append((combo, mismatches))
            if mismatches >= 0:
                total += mismatches
        except Exception as e:
            print(f"[FATAL] {key}: {e}")

    os.makedirs(base_args.log_root, exist_ok=True)
    summary_path = os.path.join(base_args.log_root, "summary.txt")
    with open(summary_path, "w") as f:
        for combo, cnt in results:
            line = f"{combo}: ERROR" if cnt == -1 else f"{combo}: {cnt} mismatches"
            f.write(line + "\n")
        f.write(f"TOTAL mismatches across datasets: {total}\n")
    print(f"[OK] Wrote summary -> {summary_path}")

if __name__ == "__main__":
    main()
