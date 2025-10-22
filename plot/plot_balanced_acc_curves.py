import json
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def ckp_num(k: str) -> int:
    """Parse keys like 'ckp_6' -> 6"""
    try:
        return int(k.split("_")[1])
    except Exception:
        return -1

def plot_balanced_accuracy_curves(json_path: str, out_path: str, title_suffix: str = "", title_prefix: str = ""):
    """
    Plot balanced accuracy curves:
    - Main curve: Continue training (data[ckp][ckp] balanced accuracy)
    - Stop at 1/3: Flat line from 1/3 checkpoint onward
    - Stop at 2/3: Flat line from 2/3 checkpoint onward
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Get all checkpoint keys and sort them
    ckp_keys = [k for k in data.keys() if k.startswith('ckp_') and k != 'average']
    ckp_keys_sorted = sorted(ckp_keys, key=ckp_num)
    
    # Extract main curve data (data[ckp][ckp] balanced accuracy)
    xs = []
    ys = []
    
    for ckp in ckp_keys_sorted:
        if ckp in data and ckp in data[ckp]:
            x = ckp_num(ckp)
            y = data[ckp][ckp]["balanced_accuracy"]
            xs.append(x)
            ys.append(y)
    
    if len(xs) == 0:
        print(f"No valid data found in {json_path}")
        return
    
    n = len(ys)
    c1 = max(1, math.ceil(n/3))  # 1/3 checkpoint index
    c2 = max(1, math.ceil(2*n/3))  # 2/3 checkpoint index
    
    # Get the checkpoint keys for 1/3 and 2/3 positions
    ckp_1_3 = ckp_keys_sorted[c1-1] if c1 <= len(ckp_keys_sorted) else None
    ckp_2_3 = ckp_keys_sorted[c2-1] if c2 <= len(ckp_keys_sorted) else None
    # import pdb; pdb.set_trace()
    # Extract balanced accuracy curves for 1/3 and 2/3 checkpoints
    xs_1_3, ys_1_3 = [], []
    xs_2_3, ys_2_3 = [], []
    
    # For 1/3 checkpoint curve
    if ckp_1_3 and ckp_1_3 in data:
        for ckp_key in ckp_keys_sorted:
            if ckp_key in data[ckp_1_3]:
                xs_1_3.append(ckp_num(ckp_key))
                ys_1_3.append(data[ckp_1_3][ckp_key]["balanced_accuracy"])
    
    # For 2/3 checkpoint curve
    if ckp_2_3 and ckp_2_3 in data:
        for ckp_key in ckp_keys_sorted:
            if ckp_key in data[ckp_2_3]:
                xs_2_3.append(ckp_num(ckp_key))
                ys_2_3.append(data[ckp_2_3][ckp_key]["balanced_accuracy"])

    # Create the plot
    plt.figure(figsize=(10, 6), dpi=140)
    
    # Main curve
    plt.plot(xs, ys, marker="o", linewidth=2.5, markersize=6, 
             label=f"Continue training till the end", color='#2E86AB', alpha=0.9)
    
    # 1/3 and 2/3 curves
    if xs_1_3 and ys_1_3:
        plt.plot(xs_1_3, ys_1_3, linestyle="--", marker="s", linewidth=2, markersize=5,
                 label=f"Stop at 33% checkpoint ({ckp_1_3})", 
                 color='#A23B72', alpha=0.8)
    if xs_2_3 and ys_2_3:
        plt.plot(xs_2_3, ys_2_3, linestyle="--", marker="^", linewidth=2, markersize=5,
                 label=f"Stop at 67% checkpoint ({ckp_2_3})", 
                 color='#F18F01', alpha=0.8)
    
    # Add vertical lines at cutoff points
    if c1 <= len(xs):
        plt.axvline(x=xs[c1-1], linestyle=":", linewidth=1.5, color='#A23B72', alpha=0.6)
    if c2 <= len(xs):
        plt.axvline(x=xs[c2-1], linestyle=":", linewidth=1.5, color='#F18F01', alpha=0.6)
    
    # Annotate the gaps at the final checkpoint
    if len(xs) > 0:
        x_final = xs[-1]
        y_final = ys[-1]
        y_s1 = ys_1_3[-1] if ys_1_3 and xs_1_3 and xs_1_3[-1] == x_final else None
        y_s2 = ys_2_3[-1] if ys_2_3 and xs_2_3 and xs_2_3[-1] == x_final else None
    
    # Formatting
    plt.title(f"{title_prefix} Continue vs. Early Stop {title_suffix}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Checkpoint Index", fontsize=12)
    plt.ylabel("Accuracy per category", fontsize=12)
    plt.xticks(xs, fontsize=10)
    
    # Set y-axis limits with padding
    valid_ys = ys.copy()
    if ys_1_3:
        valid_ys.extend(ys_1_3)
    if ys_2_3:
        valid_ys.extend(ys_2_3)
    if valid_ys:
        lo = min(valid_ys)
        hi = max(valid_ys)
        pad = (hi - lo) * 0.08 if hi > lo else 0.05
        plt.ylim(lo - pad, hi + pad)
    
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Saved balanced accuracy plot: {out_path}")

datasets = [
    "orinoquia/orinoquia_N25",
    "na/na_archbold_FL-32",
    "serengeti/serengeti_F05",
    "serengeti/serengeti_E08",
    "MAD/MAD_MAD01",
    "serengeti/serengeti_K13",
    "serengeti/serengeti_T13",
    "nz/nz_EFH_HCAMG12",
    "serengeti/serengeti_Q13",
    "serengeti/serengeti_F13",
    "nz/nz_EFH_HCAMD07",
    "nz/nz_EFD_DCAMF09",
    "serengeti/serengeti_D12",
    "MAD/MAD_A05",
    "caltech/caltech_88",
    "serengeti/serengeti_L09",
    "serengeti/serengeti_E03",
    "KGA/KGA_KHOGB07",
    "nz/nz_EFH_HCAMC01",
    "KGA/KGA_KHOLA08",
    "orinoquia/orinoquia_N29",
    "nz/nz_EFH_HCAMD03",
    "nz/nz_EFH_HCAMB10",
    "KGA/KGA_KHOLA03",
    "nz/nz_PS1_CAM6213",
    "nz/nz_EFD_DCAMB02"
]

def main():
    for ds in datasets:
        ds = ds.replace("/", "_", 1)
        json_path = f"/fs/scratch/PAS2099/camera-trap-final/accu_eval_logs/{ds}/accu_eval_lora_bsm/bioclip2/lora_8_text_head_merge_factor_1/log/eval_only_summary.json"
        out_path = f"/users/PAS2099/mino/ICICLE/plots/balanced_acc_curves_{ds}.png"

        if os.path.exists(json_path):
            plot_balanced_accuracy_curves(json_path, out_path, title_suffix=f" - {ds}", title_prefix="best accum")

if __name__ == "__main__":
    main()
