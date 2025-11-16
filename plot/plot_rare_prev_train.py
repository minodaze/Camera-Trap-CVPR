import json
import os
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

# Reuse the rare dataset list
RARE_LIST_PATH = '/users/PAS2099/mino/ICICLE/uselist/rare.txt'
with open(RARE_LIST_PATH, 'r') as f:
    RARE_DATASETS = [line.strip() for line in f if line.strip()]

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Rare (1).csv')

# Root pattern paths (adjust if directory changes)
DATA_ROOT = '/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare'
PRED_ROOT = '/fs/ess/PAS2099/camera-trap-CVPR-logs/rare_best_accum'
SPLIT_VERSION = '30'  # version folder name used in existing script

OUT_DIR = os.path.join('plots2', 'rare_prev_train')
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, 'r') as f:
        return json.load(f)


def sorted_ckp_keys(d: Dict[str, Any]) -> List[str]:
    keys = [k for k in d.keys() if k.startswith('ckp_')]
    keys.sort(key=lambda s: int(s.split('_')[-1]))
    return keys


def generate_distinct_colors(n: int, seed: int = 42) -> List[Tuple[float, float, float]]:
    import random, colorsys
    if n <= 0:
        return []
    random.seed(seed)
    hues = [i / n for i in range(n)]
    random.shuffle(hues)
    colors: List[Tuple[float, float, float]] = []
    base_sat = 0.65
    base_val = 0.92
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, base_sat, base_val)
        colors.append((r, g, b))
    if n > 50:
        for i in range(n):
            r, g, b = colors[i]
            mod = i % 3
            if mod == 1:
                colors[i] = (r * 0.85, g * 0.85, b * 0.85)
            elif mod == 2:
                colors[i] = (
                    1 - (1 - r) * 0.5,
                    1 - (1 - g) * 0.5,
                    1 - (1 - b) * 0.5,
                )
    return colors


def build_matrix(list_of_dicts: List[Dict[str, int]], classes: List[str]):
    import numpy as np
    m = np.zeros((len(classes), len(list_of_dicts)), dtype=int)
    for j, d in enumerate(list_of_dicts):
        for i, cls in enumerate(classes):
            m[i, j] = d.get(cls, 0)
    return m


def collect_train_classes_per_ckp(train_dist: Dict[str, List[Dict[str, Any]]]) -> Dict[str, set]:
    per_ckp = {}
    for ckp, samples in train_dist.items():
        if not ckp.startswith('ckp_'):
            continue
        labels = {s.get('common') for s in samples if 'common' in s}
        per_ckp[ckp] = labels
    return per_ckp


def count_rare_prev_train(train_dist: Dict[str, List[Dict[str, Any]]],
                          rare_dist: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[Dict[str, int]]]:
    """For each checkpoint i, count rare samples in ckp_i whose class appeared in ANY previous train checkpoint (0..i-1).

    The first checkpoint has no previously seen classes → empty counts.
    Checkpoint i uses cumulative union of train classes from earlier checkpoints.
    """
    ckp_keys = sorted_ckp_keys(train_dist)
    rare_keys = sorted_ckp_keys(rare_dist)
    ordered = [k for k in ckp_keys if k in rare_keys]
    train_classes = collect_train_classes_per_ckp(train_dist)
    result_counts: List[Dict[str, int]] = []
    cumulative_seen: set = set()
    for ckp in ordered:
        # Classes seen before processing this checkpoint (exclude current ckp's train classes)
        prev_classes = set(cumulative_seen)
        counts: Dict[str, int] = {}
        if prev_classes:
            for sample in rare_dist.get(ckp, []):
                label = sample.get('common')
                if label in prev_classes:
                    counts[label] = counts.get(label, 0) + 1
        # Append counts for this checkpoint
        result_counts.append(counts)
        # Update cumulative with current train classes
        cumulative_seen |= train_classes.get(ckp, set())
    return ordered, result_counts


def count_prevtrain_for_current_rare(train_dist: Dict[str, List[Dict[str, Any]]],
                                     rare_dist: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[Dict[str, int]]]:
    """For each checkpoint i>0, count samples in previous train checkpoint (i-1)
    restricted to classes that appear in current rare checkpoint i.

    First checkpoint has no previous → empty counts.
    """
    ordered = [k for k in sorted_ckp_keys(train_dist) if k in rare_dist]
    counts_per_ckp: List[Dict[str, int]] = []
    for idx, ckp in enumerate(ordered):
        if idx == 0:
            counts_per_ckp.append({})
            continue
        prev_ckp = ordered[idx - 1]
        rare_classes_now = {s.get('common') for s in rare_dist.get(ckp, []) if 'common' in s}
        counts: Dict[str, int] = {}
        for sample in train_dist.get(prev_ckp, []):
            lbl = sample.get('common')
            if lbl in rare_classes_now:
                counts[lbl] = counts.get(lbl, 0) + 1
        counts_per_ckp.append(counts)
    return ordered, counts_per_ckp


def count_cumtrain_for_current_rare(train_dist: Dict[str, List[Dict[str, Any]]],
                                    rare_dist: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[Dict[str, int]]]:
    """For each checkpoint i, compute cumulative TRAIN counts up to and including ckp i,
    but keep only classes that appear in the CURRENT rare checkpoint i.

    This answers: when examining rare ckp i, how many train samples (so far) exist for the
    classes that appear in this rare ckp?
    """
    ordered = [k for k in sorted_ckp_keys(train_dist) if k in rare_dist]
    cumulative: Dict[str, int] = {}
    out: List[Dict[str, int]] = []
    for ckp in ordered:
        # Filter CURRENT rare classes against cumulative train counts so far (exclusive of current ckp)
        rare_classes_now = {s.get('common') for s in rare_dist.get(ckp, []) if 'common' in s}
        filt = {c: cumulative.get(c, 0) for c in rare_classes_now if cumulative.get(c, 0) > 0}
        out.append(filt)
        # Now update cumulative with current train checkpoint samples (for next iterations)
        for s in train_dist.get(ckp, []):
            lbl = s.get('common')
            if lbl is not None:
                cumulative[lbl] = cumulative.get(lbl, 0) + 1
    return ordered, out


def compute_prev_train_coverage_ratio(train_dist: Dict[str, List[Dict[str, Any]]],
                                      rare_dist: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[float]]:
    """For each checkpoint i, compute coverage ratio using cumulative previous train checkpoints (0..i-1):
    ratio_i = | rare_classes(i) ∩ union_train_classes(0..i-1) | / | rare_classes(i) |

    If there are no classes in rare ckp i, the ratio is 0.0.
    Returns ordered checkpoints and list of ratios aligned with them.
    """
    ordered = [k for k in sorted_ckp_keys(train_dist) if k in rare_dist]
    train_classes = collect_train_classes_per_ckp(train_dist)
    ratios: List[float] = []
    cumulative_seen: set = set()
    for ckp in ordered:
        rare_classes_now = {s.get('common') for s in rare_dist.get(ckp, []) if 'common' in s}
        denom = len(rare_classes_now)
        if denom == 0:
            ratios.append(0.0)
        else:
            num = len(rare_classes_now & cumulative_seen)
            ratios.append(num / denom)
        # update cumulative with current train classes for next checkpoints
        cumulative_seen |= train_classes.get(ckp, set())
    return ordered, ratios


def plot_rare_prev_train(ds: str):
    ds_norm = ds.replace('/', '_')
    train_path = f"{DATA_ROOT}/{ds}/{SPLIT_VERSION}/train.json"
    rare_path = f"{DATA_ROOT}/{ds}/{SPLIT_VERSION}/rare.json"
    if not (os.path.exists(train_path) and os.path.exists(rare_path)):
        print(f"[WARN] Missing files for {ds}; skipping")
        return
    train_dist = load_json(train_path)
    rare_dist = load_json(rare_path)

    # Top: rare counts for classes seen in ANY previous train checkpoints
    ckp_order, counts_top = count_rare_prev_train(train_dist, rare_dist)
    # Bottom: cumulative train counts (up to current ckp) restricted by classes appearing in current rare
    ckp_order_prev, counts_prev = count_cumtrain_for_current_rare(train_dist, rare_dist)

    # Align checkpoint order if needed
    if ckp_order_prev != ckp_order:
        # Build a map for prev and reorder to match ckp_order
        prev_map = {ckp: counts_prev[i] for i, ckp in enumerate(ckp_order_prev)}
        counts_prev = [prev_map.get(k, {}) for k in ckp_order]

    # Global class list across both panels
    all_classes = sorted({c for d in counts_top for c in d.keys()} | {c for d in counts_prev for c in d.keys()})
    if not all_classes:
        print(f"[INFO] No overlapping rare classes with previous train checkpoints for {ds}")
        return
    colors = generate_distinct_colors(len(all_classes))
    color_map = {cls: colors[i] for i, cls in enumerate(all_classes)}

    import numpy as np
    mat_top = build_matrix(counts_top, all_classes)
    mat_prev = build_matrix(counts_prev, all_classes)
    x = np.arange(len(ckp_order))
    x_labels = [k.split('_')[-1] for k in ckp_order]

    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    def plot_stacked(ax, mat, title):
        bottom = np.zeros(mat.shape[1], dtype=int)
        for i, cls in enumerate(all_classes):
            row = mat[i]
            if row.sum() == 0:
                continue
            ax.bar(x, row, bottom=bottom, color=color_map[cls], edgecolor='black', linewidth=0.4, label=cls)
            bottom += row
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

    # Top subplot: cumulative previous-train seen classes in rare
    plot_stacked(axes[0], mat_top, f"{ds} - Rare counts (classes seen in ANY previous train checkpoints)")
    # Bottom subplot: cumulative train counts up to previous ckp (<= i-1), filtered by current rare classes
    plot_stacked(axes[1], mat_prev, f"{ds} - Cumulative train counts (<= ckp i-1) for classes in current rare")

    # Third subplot: ratio of rare classes at ckp i that appeared in previous train ckp (i-1)
    ckp_order_ratio, ratios = compute_prev_train_coverage_ratio(train_dist, rare_dist)
    # Align ratios order to ckp_order
    if ckp_order_ratio != ckp_order:
        rmap = {ckp: ratios[i] for i, ckp in enumerate(ckp_order_ratio)}
        ratios = [rmap.get(k, 0.0) for k in ckp_order]
    axes[2].plot(x, ratios, marker='o', color='tab:blue', linewidth=1.8)
    axes[2].set_title(f"{ds} - Ratio of rare classes covered by previous train (<= ckp i-1)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel('Ratio')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_xlabel('Checkpoint')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels)

    # Legend configuration: place outside right if many classes
    ncol = 1
    if len(all_classes) > 30:
        ncol = 4
    elif len(all_classes) > 20:
        ncol = 3
    elif len(all_classes) > 10:
        ncol = 2

    # One global legend for both subplots
    handles = [plt.Rectangle((0,0),1,1, facecolor=color_map[c], edgecolor='black', linewidth=0.4) for c in all_classes]
    labels = list(all_classes)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=ncol, title='Classes')
    fig.tight_layout(rect=(0,0,0.86,1))

    out_path = os.path.join(OUT_DIR, f"{ds_norm}_rare_prev_train.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def main():
    for ds in RARE_DATASETS:
        plot_rare_prev_train(ds)


if __name__ == '__main__':
    main()
