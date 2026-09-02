"""
Compare balanced accuracy curves across up to three prediction/summary JSON files.

Supports two JSON formats produced during training:
  • final_image_level_predictions.json  – flat keys: ckp_N.balanced_accuracy
  • final_training_summary.json         – nested:   checkpoint_results.ckp_N.balanced_accuracy

Usage:
    python scripts/compare_ckp_accuracy.py \
        --json  /path/to/file1.json  "Label 1" \
        --json  /path/to/file2.json  "Label 2" \
        --json  /path/to/file3.json  "Label 3" \
        [--out  ./output_dir]        \
        [--metric balanced_accuracy] \
        [--title "My comparison"]

--metric can be: balanced_accuracy | accuracy | loss
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_ckp_metric(data: dict, metric: str) -> tuple[list[str], list[float]]:
    """
    Return (ckp_keys_sorted, values) for the requested metric.
    Handles both flat and nested (checkpoint_results) formats.
    """
    # Detect nested format
    if "checkpoint_results" in data:
        ckp_dict = data["checkpoint_results"]
    else:
        ckp_dict = {k: v for k, v in data.items() if k.startswith("ckp_")}

    keys = sorted(ckp_dict.keys(), key=lambda k: int(k.split("_")[1]))
    values = [ckp_dict[k].get(metric, float("nan")) for k in keys]
    return keys, values


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

MARKERS = ["o", "s", "^"]
LINESTYLES = ["-", "--", ":"]
COLORS = ["steelblue", "darkorange", "green"]


def plot_comparison(
    entries: list[tuple[str, list[str], list[float]]],  # [(label, ckp_keys, values), ...]
    metric: str,
    title: str,
    out_dir: str,
):
    """Draw one figure with one curve per entry, aligned by checkpoint index."""
    fig, ax = plt.subplots(figsize=(10, 5))

    max_n = max(len(vals) for _, _, vals in entries)

    for i, (label, ckp_keys, values) in enumerate(entries):
        x = np.arange(len(values))
        ax.plot(
            x, values,
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            color=COLORS[i % len(COLORS)],
            label=label,
            linewidth=1.8,
            markersize=5,
        )
        # Annotate best point
        best_idx = int(np.nanargmax(values)) if metric != "loss" else int(np.nanargmin(values))
        best_val = values[best_idx]
        ax.annotate(
            f"{best_val:.3f}",
            xy=(x[best_idx], best_val),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=7,
            color=COLORS[i % len(COLORS)],
        )

    ax.set_xlabel("Checkpoint index")
    ax.set_xlim(-0.5, max_n - 0.5)
    ax.set_xticks(range(max_n))
    ax.set_xticklabels([f"Ckp {j+1}" for j in range(max_n)], rotation=45, ha="right")

    metric_label = metric.replace("_", " ").title()
    ax.set_ylabel(metric_label)
    if metric != "loss":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ax.set_title(title or f"{metric_label} per Checkpoint")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fname = f"compare_{metric}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare balanced-accuracy (or other metric) curves across JSON files."
    )
    parser.add_argument(
        "--json", nargs="+", action="append", metavar=("PATH", "LABEL"),
        required=True,
        help="Path to JSON file, optionally followed by a display label. "
             "Repeat up to 3 times, e.g.: --json file1.json 'Run A' --json file2.json 'Run B'",
    )
    parser.add_argument(
        "--metric", default="balanced_accuracy",
        choices=["balanced_accuracy", "accuracy", "loss"],
        help="Metric to plot (default: balanced_accuracy)",
    )
    parser.add_argument(
        "--title", default="/users/PAS2099/mino/ICICLE/plots", help="Custom plot title"
    )
    parser.add_argument(
        "--out", default="./comparison_plots", help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.json) > 3:
        raise SystemExit("At most 3 --json arguments are supported.")

    os.makedirs(args.out, exist_ok=True)

    entries = []
    for spec in args.json:
        path  = spec[0]
        label = spec[1] if len(spec) > 1 else os.path.basename(os.path.dirname(path))
        print(f"Loading {path}  (label: {label}) ...")
        data = load_json(path)
        ckp_keys, values = extract_ckp_metric(data, args.metric)
        print(f"  → {len(ckp_keys)} checkpoints, "
              f"best {args.metric} = {max(v for v in values if not np.isnan(v)):.4f}")
        entries.append((label, ckp_keys, values))

    plot_comparison(entries, args.metric, args.title, args.out)
    print(f"\nDone. Output written to: {args.out}")


if __name__ == "__main__":
    main()
