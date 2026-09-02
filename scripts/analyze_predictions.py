"""
Visualize and analyze final_image_level_predictions.json produced during training.

Usage:
    python scripts/analyze_predictions.py \
        --json /path/to/final_image_level_predictions.json \
        [--out  ./output_dir]

Produces:
  1. accuracy_curve.png       – accuracy & balanced-accuracy over checkpoints
  2. confidence_curve.png     – avg / true / false confidence over checkpoints
  3. loss_curve.png           – loss over checkpoints (if available)
  4. class_distribution.png   – ground-truth class distribution
  5. predicted_dist_per_ckp.png – predicted class distribution per checkpoint
  6. confusion_matrix_per_ckp.png – grid of confusion matrices for every checkpoint
  6b. confusion_matrix_best.png   – standalone confusion matrix for the best checkpoint
  7. per_class_accuracy.png   – per-class accuracy heat-map over checkpoints
  8. confidence_hist.png      – confidence histogram (correct vs incorrect)
  9. summary.txt              – text summary of key metrics
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def checkpoint_keys(data: dict) -> list[str]:
    """Return sorted ckp_N keys."""
    keys = [k for k in data if k.startswith("ckp_")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def class_from_label_key(label_key: str) -> str:
    """'True label: springbok' -> 'springbok'"""
    return label_key.replace("True label: ", "").strip()


def build_per_ckp_data(data: dict, ckp_keys: list[str]) -> dict:
    """Extract per-checkpoint scalar metrics and per-class correct/incorrect counts."""
    metrics = defaultdict(list)          # metric_name -> [val per ckp]
    per_class_correct  = defaultdict(list)   # class -> [count per ckp]
    per_class_total    = defaultdict(list)   # class -> [count per ckp]
    all_confidences    = {"correct": [], "incorrect": []}  # pooled across ckps

    classes = list(data["stats"]["class_dist"].keys())

    for ckp in ckp_keys:
        ck = data[ckp]
        metrics["accuracy"].append(ck.get("accuracy", float("nan")))
        metrics["balanced_accuracy"].append(ck.get("balanced_accuracy", float("nan")))
        metrics["avg_confidence"].append(ck.get("avg_confidence", float("nan")))
        metrics["true_confidence"].append(ck.get("true_confidence", float("nan")))
        metrics["false_confidence"].append(ck.get("false_confidence", float("nan")))
        metrics["loss"].append(ck.get("loss", float("nan")))

        # per-class correct / total
        correct_dict   = ck.get("correct",   {})
        incorrect_dict = ck.get("incorrect", {})

        for cls in classes:
            key = f"True label: {cls}"
            c_list = correct_dict.get(key, [])
            i_list = incorrect_dict.get(key, [])
            per_class_correct[cls].append(len(c_list))
            per_class_total[cls].append(len(c_list) + len(i_list))

            for item in c_list:
                all_confidences["correct"].append(item["confidence"])
            for item in i_list:
                all_confidences["incorrect"].append(item["confidence"])

    return metrics, per_class_correct, per_class_total, all_confidences


def per_class_accuracy_matrix(
    classes: list[str],
    per_class_correct: dict,
    per_class_total: dict,
    n_ckp: int,
) -> np.ndarray:
    """Returns shape (n_classes, n_ckp) with accuracy values (NaN when no samples)."""
    mat = np.full((len(classes), n_ckp), np.nan)
    for i, cls in enumerate(classes):
        for j in range(n_ckp):
            tot = per_class_total[cls][j]
            if tot > 0:
                mat[i, j] = per_class_correct[cls][j] / tot
    return mat


def build_confusion_matrix(data: dict, ckp_key: str, classes: list[str]) -> np.ndarray:
    """Build confusion matrix for one checkpoint. Rows = true, cols = predicted."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)

    ck = data[ckp_key]
    correct_dict   = ck.get("correct",   {})
    incorrect_dict = ck.get("incorrect", {})

    for label_key, items in correct_dict.items():
        true_cls = class_from_label_key(label_key)
        if true_cls in idx:
            ti = idx[true_cls]
            cm[ti, ti] += len(items)

    for label_key, items in incorrect_dict.items():
        true_cls = class_from_label_key(label_key)
        if true_cls not in idx:
            continue
        ti = idx[true_cls]
        for item in items:
            pred_cls = item.get("prediction", "")
            if pred_cls in idx:
                pi = idx[pred_cls]
                cm[ti, pi] += 1

    return cm


# ──────────────────────────────────────────────────────────────────────────────
# Plot functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_accuracy_curve(metrics: dict, ckp_labels: list[str], out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(ckp_labels))
    ax.plot(x, metrics["accuracy"],          "o-", label="Accuracy")
    ax.plot(x, metrics["balanced_accuracy"], "s--", label="Balanced Accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ckp_labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Accuracy over Checkpoints")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)


def plot_confidence_curve(metrics: dict, ckp_labels: list[str], out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(ckp_labels))
    ax.plot(x, metrics["avg_confidence"],   "o-",  label="Avg confidence")
    ax.plot(x, metrics["true_confidence"],  "^-",  label="Correct confidence", color="green")
    ax.plot(x, metrics["false_confidence"], "v--", label="Incorrect confidence", color="red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ckp_labels, rotation=45, ha="right")
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Confidence over Checkpoints")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_curve.png"), dpi=150)
    plt.close(fig)


def plot_loss_curve(metrics: dict, ckp_labels: list[str], out_dir: str):
    losses = metrics["loss"]
    if all(np.isnan(v) for v in losses):
        return  # no loss data
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(ckp_labels))
    ax.plot(x, losses, "o-", color="purple", label="Loss")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ckp_labels, rotation=45, ha="right")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Loss over Checkpoints")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


def plot_class_distribution(
    class_dist: dict,
    per_class_total: dict,
    ckp_keys: list[str],
    ckp_labels: list[str],
    out_dir: str,
):
    classes = list(class_dist.keys())
    n_cls = len(classes)
    n_ckp = len(ckp_keys)
    colors = plt.cm.tab10.colors

    # Build matrix: shape (n_ckp, n_cls)
    counts = np.array([[per_class_total[cls][j] for cls in classes] for j in range(n_ckp)])

    fig, ax = plt.subplots(figsize=(max(8, n_ckp * 0.9 + 2), 5))
    x = np.arange(n_ckp)
    width = 0.8 / n_cls

    for i, cls in enumerate(classes):
        offset = (i - n_cls / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts[:, i], width, label=cls,
                      color=colors[i % len(colors)])
        for bar, val in zip(bars, counts[:, i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(val), ha="center", va="bottom", fontsize=7)

    # Add overall totals as a reference annotation
    overall = [class_dist[cls] for cls in classes]
    legend_extra = "  |  Overall: " + ", ".join(f"{cls}={cnt}" for cls, cnt in zip(classes, overall))
    ax.set_xticks(list(x))
    ax.set_xticklabels(ckp_labels, rotation=45, ha="right")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Ground-Truth Count")
    ax.set_title("Ground-Truth Class Distribution per Checkpoint")
    ax.legend(loc="upper right", fontsize=8, ncol=2, title="Class")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)


def plot_predicted_dist_per_ckp(data: dict, ckp_keys: list[str], classes: list[str], out_dir: str):
    n_ckp = len(ckp_keys)
    n_cls = len(classes)
    counts = np.zeros((n_ckp, n_cls), dtype=int)

    for j, ckp in enumerate(ckp_keys):
        class_count = data[ckp].get("class_count", {})
        for i, cls in enumerate(classes):
            counts[j, i] = class_count.get(cls, 0)

    fig, ax = plt.subplots(figsize=(max(8, n_ckp * 0.9), 5))
    x = np.arange(n_ckp)
    width = 0.8 / n_cls
    colors = plt.cm.tab10.colors
    for i, cls in enumerate(classes):
        offset = (i - n_cls / 2 + 0.5) * width
        ax.bar(x + offset, counts[:, i], width, label=cls, color=colors[i % len(colors)])

    ax.set_xticks(list(x))
    ax.set_xticklabels([k.replace("ckp_", "Ckp ") for k in ckp_keys], rotation=45, ha="right")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Predicted Count")
    ax.set_title("Predicted Class Distribution per Checkpoint")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "predicted_dist_per_ckp.png"), dpi=150)
    plt.close(fig)


def _draw_confusion_matrix(ax, cm: np.ndarray, classes: list[str], title: str):
    """Draw a single confusion matrix onto an existing Axes."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    return im


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], ckp_label: str,
                          out_dir: str, filename: str = "confusion_matrix.png"):
    """Standalone confusion matrix for a single checkpoint."""
    cell = max(6, len(classes))
    fig, ax = plt.subplots(figsize=(cell, cell - 1))
    im = _draw_confusion_matrix(ax, cm, classes, f"Confusion Matrix – {ckp_label}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


def plot_all_confusion_matrices(
    data: dict, ckp_keys: list[str], ckp_labels: list[str],
    classes: list[str], out_dir: str
):
    """Grid of confusion matrices – one cell per checkpoint."""
    n = len(ckp_keys)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    cell = max(3, len(classes))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell, nrows * cell))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for idx, (ckp, label) in enumerate(zip(ckp_keys, ckp_labels)):
        cm = build_confusion_matrix(data, ckp, classes)
        im = _draw_confusion_matrix(axes[idx], cm, classes, label)
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Confusion Matrix per Checkpoint", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_per_ckp.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(
    mat: np.ndarray, classes: list[str], ckp_labels: list[str], out_dir: str
):
    fig, ax = plt.subplots(figsize=(max(8, len(ckp_labels) * 0.9), max(4, len(classes) * 0.6)))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(range(len(ckp_labels)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(ckp_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Class")
    ax.set_title("Per-Class Accuracy over Checkpoints")

    for i in range(len(classes)):
        for j in range(len(ckp_labels)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_confidence_hist(all_confidences: dict, out_dir: str):
    c_vals = all_confidences["correct"]
    i_vals = all_confidences["incorrect"]
    if not c_vals and not i_vals:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 30
    if c_vals:
        ax.hist(c_vals, bins=bins, alpha=0.6, label=f"Correct (n={len(c_vals)})",
                color="green", density=True)
    if i_vals:
        ax.hist(i_vals, bins=bins, alpha=0.6, label=f"Incorrect (n={len(i_vals)})",
                color="red", density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Correct vs Incorrect (all checkpoints)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_hist.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary text
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(data: dict, metrics: dict, ckp_keys: list[str],
                  per_class_correct: dict, per_class_total: dict,
                  classes: list[str], out_dir: str):
    stats = data["stats"]
    lines = []
    lines.append("=" * 60)
    lines.append("TRAINING PREDICTION ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Number of classes     : {stats['num_cls']}")
    lines.append(f"Total samples         : {stats['total_samples']}")
    lines.append(f"Total checkpoints     : {stats['total_checkpoints']}")
    lines.append("")
    lines.append("Average metrics across all checkpoints:")
    avg = stats.get("average", {})
    lines.append(f"  Accuracy            : {avg.get('accuracy', float('nan')):.4f}")
    lines.append(f"  Balanced Accuracy   : {avg.get('balanced_accuracy', float('nan')):.4f}")
    lines.append(f"  Avg Confidence      : {avg.get('average_confidence', float('nan')):.4f}")
    lines.append(f"  Loss                : {avg.get('loss', float('nan')):.4f}")
    lines.append("")

    # Best checkpoint by balanced accuracy
    ba = metrics["balanced_accuracy"]
    if any(not np.isnan(v) for v in ba):
        best_idx = int(np.nanargmax(ba))
        lines.append(f"Best checkpoint (balanced acc): {ckp_keys[best_idx]}"
                     f"  →  BA={ba[best_idx]:.4f}, Acc={metrics['accuracy'][best_idx]:.4f}")
    lines.append("")

    lines.append("Ground-truth class distribution:")
    for cls, cnt in stats["class_dist"].items():
        pct = cnt / stats["total_samples"] * 100
        lines.append(f"  {cls:<25s} {cnt:4d}  ({pct:.1f}%)")
    lines.append("")

    lines.append("Per-class accuracy at each checkpoint:")
    header = f"{'Class':<25s}" + "".join(f"{k:>8s}" for k in ckp_keys)
    lines.append(header)
    for cls in classes:
        row = f"{cls:<25s}"
        for j in range(len(ckp_keys)):
            tot = per_class_total[cls][j]
            if tot > 0:
                acc = per_class_correct[cls][j] / tot
                row += f"{acc:8.3f}"
            else:
                row += "     N/A"
        lines.append(row)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze final_image_level_predictions.json")
    parser.add_argument("--json", required=True, help="Path to final_image_level_predictions.json")
    parser.add_argument("--out",  default="/users/PAS2099/mino/ICICLE/plots/prediction_KGA_zs30", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.json} ...")
    data = load_json(args.json)

    ckp_keys   = checkpoint_keys(data)
    ckp_labels = [k.replace("ckp_", "Ckp ") for k in ckp_keys]
    classes    = list(data["stats"]["class_dist"].keys())
    n_ckp      = len(ckp_keys)

    print(f"  {len(classes)} classes, {n_ckp} checkpoints")

    metrics, per_class_correct, per_class_total, all_confidences = \
        build_per_ckp_data(data, ckp_keys)

    print("Generating plots ...")

    plot_accuracy_curve(metrics, ckp_labels, args.out)
    print("  Saved: accuracy_curve.png")

    plot_confidence_curve(metrics, ckp_labels, args.out)
    print("  Saved: confidence_curve.png")

    plot_loss_curve(metrics, ckp_labels, args.out)

    plot_class_distribution(
        data["stats"]["class_dist"], per_class_total, ckp_keys, ckp_labels, args.out
    )
    print("  Saved: class_distribution.png")

    plot_predicted_dist_per_ckp(data, ckp_keys, classes, args.out)
    print("  Saved: predicted_dist_per_ckp.png")

    plot_all_confusion_matrices(data, ckp_keys, ckp_labels, classes, args.out)
    print("  Saved: confusion_matrix_per_ckp.png")

    best_idx = int(np.nanargmax(metrics["balanced_accuracy"]))
    cm_best = build_confusion_matrix(data, ckp_keys[best_idx], classes)
    plot_confusion_matrix(cm_best, classes, ckp_labels[best_idx], args.out,
                          filename="confusion_matrix_best.png")
    print(f"  Saved: confusion_matrix_best.png  ({ckp_labels[best_idx]})")

    acc_mat = per_class_accuracy_matrix(classes, per_class_correct, per_class_total, n_ckp)
    plot_per_class_accuracy(acc_mat, classes, ckp_labels, args.out)
    print("  Saved: per_class_accuracy.png")

    plot_confidence_hist(all_confidences, args.out)
    print("  Saved: confidence_hist.png")

    write_summary(data, metrics, ckp_keys, per_class_correct, per_class_total, classes, args.out)

    print(f"\nDone. All outputs written to: {args.out}")


if __name__ == "__main__":
    main()
