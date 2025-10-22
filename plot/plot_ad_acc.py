import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

def ckp_num(k: str) -> int:
    try:
        return int(k.split("_")[1])
    except Exception:
        return -1

def extract_curve(block: dict, metric: str):
    keys_sorted = sorted(block.keys(), key=ckp_num)
    xs = [ckp_num(k) for k in keys_sorted]
    ys = [block[k][metric] for k in keys_sorted]
    return xs, ys

def build_plot(json_path: str, out_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    top_keys_sorted = sorted(data.keys(), key=ckp_num)
    last_key = top_keys_sorted[-1]
    last_block = data[last_key]

    xs_main, ys_main = extract_curve(last_block, "accuracy")
    n = len(xs_main)

    c1 = math.ceil(n/3)
    c2 = math.ceil(2*n/3)

    def starts_at_balacc(ckp_index: int):
        top_key = f"ckp_{ckp_index}"
        block = data[top_key]
        xs, ys = extract_curve(block, "balanced_accuracy")
        ys_masked = [y if x >= ckp_index else float("nan") for x, y in zip(xs, ys)]
        return xs, ys_masked

    xs1, ys1 = starts_at_balacc(c1)
    xs2, ys2 = starts_at_balacc(c2)

    plt.figure(figsize=(9, 5.5), dpi=140)
    plt.plot(xs_main, ys_main, marker="o", linewidth=2, label=f"Train all")
    plt.plot(xs1, ys1, linestyle="--", marker="s", label=f"From 1/3 (ckp {c1}) — balanced acc.")
    plt.plot(xs2, ys2, linestyle="--", marker="^", label=f"From 2/3 (ckp {c2}) — balanced acc.")

    for c in (c1, c2):
        plt.axvline(x=c, linestyle=":", linewidth=1)

    plt.title("Accuracy (last block) + Balanced-Accuracy from 1/3 & 2/3 starts")
    plt.xlabel("Checkpoint index")
    plt.ylabel("Accuracy per checkpoint")
    plt.xticks(xs_main)

    import numpy as np
    lo_candidates = [v for v in ys_main if not np.isnan(v)] + [v for v in ys1 if not np.isnan(v)] + [v for v in ys2 if not np.isnan(v)]
    lo, hi = min(lo_candidates), max(lo_candidates)
    pad = (hi - lo) * 0.06 if hi > lo else 0.02
    plt.ylim(lo - pad, hi + pad)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out_path}")

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

if __name__ == "__main__":
    json_temp = '/fs/scratch/PAS2099/camera-trap-final/eval_logs/{ds}/eval_full_accu/bioclip2/full_text_head_merge_factor_1/log/eval_only_summary.json'
    for ds in datasets:
        ds = ds.replace('/', '_', 1)
        json_path = json_temp.format(ds=ds)
        out_path = f"/users/PAS2099/mino/ICICLE/plots/ad_acc_{ds.replace('/', '_', 1)}_stops.png"
        # import pdb; pdb.set_trace()
        if os.path.exists(json_path):
            build_plot(json_path, out_path)
        else:
            # /fs/scratch/PAS2099/camera-trap-final/eval_logs/serengeti_serengeti_F13/eval_full_accu/bioclip2/full_text_head_merge_factor_1/log/eval_only_summary.json
            print(f"JSON file not found: {json_path}")
