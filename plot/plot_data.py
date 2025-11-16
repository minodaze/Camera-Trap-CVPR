import json
import os
from typing import List, Dict, Any, Tuple, Optional
import argparse
import matplotlib.pyplot as plt

with open('/users/PAS2099/mino/ICICLE/uselist/rare.txt', 'r') as f:
    rare_datasets = [line.strip() for line in f.readlines()]

def parse_args():
    parser = argparse.ArgumentParser(description="Load and process JSON data.")
    parser.add_argument("--train", type=str, help="Path to the train data JSON file.")
    parser.add_argument("--test", type=str, help="Path to the test data JSON file.")
    return parser.parse_args()

def load_data_dist_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_pred_json_data(file_path: str):
    """Load prediction json (with correct/incorrect per checkpoint)."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def _collect_id_sets(rare_dist: Dict[str, List[Dict[str, Any]]],
                     test_dist: Dict[str, List[Dict[str, Any]]]) -> Tuple[set, set]:
    """Collect union sets of sample IDs for rare and test splits across all ckps."""
    rare_ids, test_ids = set(), set()
    for _, samples in rare_dist.items():
        if not isinstance(samples, list):
            continue
        for s in samples:
            sid = _extract_sample_id(s)
            if sid:
                rare_ids.add(sid)
    for _, samples in test_dist.items():
        if not isinstance(samples, list):
            continue
        for s in samples:
            sid = _extract_sample_id(s)
            if sid:
                test_ids.add(sid)
    return rare_ids, test_ids

def _sorted_ckp_keys(d: Dict[str, Any]) -> List[str]:
    keys = [k for k in d.keys() if k.startswith('ckp_')]
    keys.sort(key=lambda s: int(s.split('_')[-1]))
    return keys

# /fs/ess/PAS2099/camera-trap-CVPR-logs/rare_best_accum/MTZ_MTZ_D03/bioclip2/lora_8_text_head/all/log

def _extract_sample_id(sample: Dict[str, Any]) -> Optional[str]:
    """Extract a stable sample identifier from a sample dict."""
    for k in ("file_path", "image_path", "img_path", "path", "image"):
        v = sample.get(k)
        if isinstance(v, str) and v:
            return v
    seq = sample.get("seq_id") or sample.get("sequence_id")
    frm = sample.get("frame_id") or sample.get("frame_num") or sample.get("frame")
    if seq is not None and frm is not None:
        return f"{seq}/{frm}"
    return None

def _generate_distinct_colors(n: int, seed: int = 42) -> List[Tuple[float, float, float]]:
    """Generate n visually distinct RGB colors.

    Strategy:
    1. Evenly spaced hues around the HSV color wheel.
    2. Fixed saturation/value chosen for good contrast on light background.
    3. For large n (>50), lightly vary brightness to reduce near-duplicate perception.
    """
    import random, colorsys
    if n <= 0:
        return []
    random.seed(seed)
    # Evenly spaced hues, then shuffle to avoid adjacent similar bars for sequential classes.
    hues = [i / n for i in range(n)]
    random.shuffle(hues)
    colors: List[Tuple[float, float, float]] = []
    base_sat = 0.65
    base_val = 0.92
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, base_sat, base_val)
        colors.append((r, g, b))
    # Light value modulation for very large palettes to increase differentiation.
    if n > 50:
        for i in range(n):
            r, g, b = colors[i]
            mod = i % 3
            if mod == 1:  # darken slightly
                colors[i] = (r * 0.85, g * 0.85, b * 0.85)
            elif mod == 2:  # lighten slightly
                colors[i] = (
                    1 - (1 - r) * 0.5,
                    1 - (1 - g) * 0.5,
                    1 - (1 - b) * 0.5,
                )
    return colors

if __name__ == "__main__":
    for ds in rare_datasets:
        ds_norm = ds.replace("/", "_")
        # Load train and test data
        train_data_dist = load_data_dist_json_data(f"/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare/{ds}/30/train.json")
        test_data_dist = load_data_dist_json_data(f"/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare/{ds}/30/test.json")
        rare_data_dist = load_data_dist_json_data(f"/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare/{ds}/30/rare.json")
        pred_data = load_pred_json_data(f"/fs/ess/PAS2099/camera-trap-CVPR-logs/rare_best_accum/{ds_norm}/bioclip2/lora_8_text_head/all/log/final_image_level_predictions.json")

        train_class_distribution = {}
        for ckp, samples in train_data_dist.items():
            for sample in samples:
                label = sample['common']
                if ckp not in train_class_distribution:
                    train_class_distribution[ckp] = {}
                if label not in train_class_distribution[ckp]:
                    train_class_distribution[ckp][label] = 0
                train_class_distribution[ckp][label] += 1
        test_data_distribution = {}
        for ckp, samples in test_data_dist.items():
            for sample in samples:
                label = sample['common']
                if ckp not in test_data_distribution:
                    test_data_distribution[ckp] = {}
                if label not in test_data_distribution[ckp]:
                    test_data_distribution[ckp][label] = 0
                test_data_distribution[ckp][label] += 1
        rare_test_data_distribution = {}
        for ckp, samples in rare_data_dist.items():
            for sample in samples:
                label = sample['common']
                if ckp not in rare_test_data_distribution:
                    rare_test_data_distribution[ckp] = {}
                if label not in rare_test_data_distribution[ckp]:
                    rare_test_data_distribution[ckp][label] = 0
                rare_test_data_distribution[ckp][label] += 1
        # Build ID sets to separate rare vs general test predictions
        rare_ids, test_ids = _collect_id_sets(rare_data_dist, test_data_dist)
        correct_preds: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        incorrect_preds: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for ckp, preds in pred_data.items():
            if not ckp.startswith('ckp_'):
                continue
            correct_preds[ckp] = preds.get('correct', {})
            incorrect_preds[ckp] = preds.get('incorrect', {})

        # Determine ckp ordering from available sources
        ckp_keys = sorted(set(_sorted_ckp_keys(train_class_distribution) +
                              _sorted_ckp_keys(test_data_distribution) +
                              _sorted_ckp_keys(rare_test_data_distribution) +
                              _sorted_ckp_keys(correct_preds) +
                              _sorted_ckp_keys(incorrect_preds)),
                          key=lambda s: int(s.split('_')[-1]))

        # 1) Train distribution: total samples per ckp (blue)
        per_ckp_train = []
        for ckp in ckp_keys:
            per_class = train_class_distribution.get(ckp, {})
            per_ckp_train.append(per_class)

        # 2) Prediction results on TEST split: correct (green) / incorrect (red)
        test_correct_per_class, test_incorrect_per_class = [], []
        # 3) Prediction results on RARE split: correct (green) / incorrect (red)
        rare_correct_per_class, rare_incorrect_per_class = [], []

        def _count_split(pred_dict: Dict[str, List[Dict[str, Any]]], id_set: set) -> int:
            cnt_per_label = {}
            for label, _ in pred_dict.items():  # grouped by True label string
                cnt_per_label[label] = cnt_per_label.get(label, 0) + 1
            return cnt_per_label

        def _strip_true_label(s: str) -> str:
            # keys look like "True label: grey rhebok"
            return s.split("True label:", 1)[-1].strip()

        test_correct_per_ckp = []
        test_incorrect_per_ckp = []
        rare_correct_per_ckp = []
        rare_incorrect_per_ckp = []
        for ckp in ckp_keys:
            c_ok = correct_preds.get(ckp, {})
            c_ng = incorrect_preds.get(ckp, {})
            # Build per-class counts (no ID filtering here because you said you already separated;
            # if not separated, you would filter items by membership in test_ids / rare_ids.)
            common_tc, rare_tc = {}, {}
            for k, items in c_ok.items():
                cls = _strip_true_label(k)
                for it in items:
                    sid = _extract_sample_id(it)
                    if sid in test_ids:
                        common_tc[cls] = common_tc.get(cls, 0) + 1
                    elif sid in rare_ids:
                        rare_tc[cls] = rare_tc.get(cls, 0) + 1

            common_ti, rare_ti = {}, {}
            for k, items in c_ng.items():
                cls = _strip_true_label(k)
                for it in items:
                    sid = _extract_sample_id(it)
                    if sid in test_ids:
                        common_ti[cls] = common_ti.get(cls, 0) + 1
                    elif sid in rare_ids:
                        rare_ti[cls] = rare_ti.get(cls, 0) + 1
            # For now treat all correct/incorrect as test; adapt if you split earlier.
            test_correct_per_ckp.append(common_tc)
            test_incorrect_per_ckp.append(common_ti)
            rare_correct_per_ckp.append(rare_tc)   # if you have separate rare sets, fill similarly
            rare_incorrect_per_ckp.append(rare_ti) # else leave empty

        # --- Unified class list across all splits ---
        all_classes = sorted(set(
            c for d in per_ckp_train for c in d.keys()
        ) | set(c for d in test_correct_per_ckp for c in d.keys())
          | set(c for d in test_incorrect_per_ckp for c in d.keys())
          | set(c for d in rare_correct_per_ckp for c in d.keys())
          | set(c for d in rare_incorrect_per_ckp for c in d.keys())
        )

        import numpy as np

        def build_matrix(list_of_dicts, classes):
            m = np.zeros((len(classes), len(list_of_dicts)), dtype=int)
            for j, d in enumerate(list_of_dicts):
                for i, cls in enumerate(classes):
                    m[i, j] = d.get(cls, 0)
            return m

        train_mat = build_matrix(per_ckp_train, all_classes)
        test_ok_mat = build_matrix(test_correct_per_ckp, all_classes)
        test_ng_mat = build_matrix(test_incorrect_per_ckp, all_classes)
        rare_ok_mat = build_matrix(rare_correct_per_ckp, all_classes)
        rare_ng_mat = build_matrix(rare_incorrect_per_ckp, all_classes)
        # --- Plot stacked bars (per checkpoint, segments = classes) ---
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
        x = np.arange(len(ckp_keys))
        x_labels = [k.split('_')[-1] for k in ckp_keys]

        def stacked_classes(ax, mat, title, colors=None, hatch=None, alpha=1.0):
            bottom = np.zeros(mat.shape[1], dtype=int)
            for i, cls in enumerate(all_classes):
                counts = mat[i]
                if counts.sum() == 0:
                    continue
                ax.bar(x, counts, bottom=bottom,
                       label=cls if title.startswith("Train") else None,
                       color=(colors[i] if colors else None),
                       hatch=hatch, alpha=alpha, linewidth=0.4, edgecolor='black')
                bottom += counts
            ax.set_title(title)
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.3)

        # Simple color cycle
        from itertools import cycle
        base_colors = cycle(plt.cm.tab20.colors)
        class_colors = [next(base_colors) for _ in all_classes]

        # Train distribution (stacked by class)
        stacked_classes(axes[0], train_mat, f"{ds} - Train class distribution", colors=class_colors)

        # --- Side-by-side stacked bars for Test and Rare ---
        import matplotlib.patches as mpatches

        def stacked_correct_incorrect_side_by_side(ax, ok_mat, ng_mat, title):
            width = 0.42
            x_ok = x - width/2
            x_ng = x + width/2

            bottom_ok = np.zeros(ok_mat.shape[1], dtype=int)
            bottom_ng = np.zeros(ng_mat.shape[1], dtype=int)

            for i, cls in enumerate(all_classes):
                ok_counts = ok_mat[i]
                ng_counts = ng_mat[i]
                if ok_counts.sum() > 0:
                    ax.bar(x_ok, ok_counts, bottom=bottom_ok, color=class_colors[i],
                           edgecolor='black', linewidth=0.3)
                if ng_counts.sum() > 0:
                    ax.bar(x_ng, ng_counts, bottom=bottom_ng, color=class_colors[i],
                           edgecolor='black', linewidth=0.3, hatch='///', alpha=0.9)
                bottom_ok += ok_counts
                bottom_ng += ng_counts

            ax.set_title(title)
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.3)

            # Legends: classes (colors) + correctness (hatch)
            class_handles = [mpatches.Patch(facecolor=class_colors[i], edgecolor='black', label=all_classes[i])
                             for i in range(len(all_classes))]
            correct_handle = mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Correct')
            incorrect_handle = mpatches.Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='Incorrect')

            # Put correctness legend inside subplot; classes legend is added globally below.
            ax.legend(handles=[correct_handle, incorrect_handle], loc='upper left', fontsize=8)

        # Test per-ckp: left = correct (stacked by class), right = incorrect (stacked by class, hatched)
        stacked_correct_incorrect_side_by_side(axes[1], test_ok_mat, test_ng_mat,
                                               "Test predictions per checkpoint (class-colored stacks)")

        # Rare per-ckp
        stacked_correct_incorrect_side_by_side(axes[2], rare_ok_mat, rare_ng_mat,
                                               "Rare predictions per checkpoint (class-colored stacks)")

        axes[2].set_xticks(x)
        axes[2].set_xticklabels(x_labels)
        axes[2].set_xlabel("Checkpoint")

        # Global legend for classes (explicit color mapping)
        import matplotlib.patches as mpatches
        # Collect rare classes for bold highlighting in legend
        rare_classes = set()
        for _ckp, lbl_map in rare_test_data_distribution.items():
            rare_classes.update(lbl_map.keys())
        class_handles = [mpatches.Patch(facecolor=class_colors[i], edgecolor='black',
                                        label=all_classes[i])
                         for i in range(len(all_classes))]
        # Arrange legend in multiple columns when many classes
        ncol = 1
        if len(all_classes) > 20:
            ncol = 3
        elif len(all_classes) > 10:
            ncol = 2
        legend = fig.legend(class_handles, [p.get_label() for p in class_handles],
                            loc='upper center', bbox_to_anchor=(0.5, 0.5),
                            fontsize=8, title="Classes", frameon=False, ncol=ncol)
        for text in legend.get_texts():
            if text.get_text() in rare_classes:
                text.set_fontweight('bold')

        plt.tight_layout(rect=(0,0,0.96,1))
        out_dir = os.path.join("plots2", "class_dist_per_ckp")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ds_norm}_per_class_ckp.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
# ...existing code...