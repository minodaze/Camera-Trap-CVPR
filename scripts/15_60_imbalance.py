import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATASET_LIST = Path("/users/PAS2099/mino/ICICLE/uselist/15_60.txt")
DEFAULT_DATA_ROOT = Path("/fs/scratch/PAS2099/camera-trap-benchmark/dataset")
DEFAULT_OUTPUT_DIR = Path("/users/PAS2099/mino/ICICLE/plots/class_imbalance")
SETUPS = [15, 30, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze class distribution imbalance across setups (15/30/60 shots) "
            "for each dataset. Computes Gini coefficient per checkpoint and writes a CSV."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Dataset names like KGA/KGA_KHOLA03. "
            "Defaults to all datasets in --dataset-list."
        ),
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        default=DEFAULT_DATASET_LIST,
        help="Text file with one dataset per line.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the summary CSV and plots will be written.",
    )
    parser.add_argument(
        "--label-type",
        default="common",
        help="Key in each image entry to use as class label (default: 'common').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save per-dataset bar plots of class counts.",
    )
    return parser.parse_args()


def load_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return [d.strip() for d in args.datasets if d.strip()]
    with args.dataset_list.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def gini_coefficient(counts: list[int]) -> float:
    """
    Gini coefficient of class sample counts.
    0 = perfectly balanced, 1 = maximally imbalanced.
    """
    arr = np.array(sorted(counts), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, arr) / (n * arr.sum())) - (n + 1) / n)


def plot_class_counts(counts: dict[str, int], title: str, out_path: Path) -> None:
    classes = list(counts.keys())
    values = list(counts.values())
    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), 4))
    ax.bar(range(len(classes)), [values[i] for i in order])
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([classes[i] for i in order], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    datasets = load_datasets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset_name in datasets:
        for setup in SETUPS:
            train_json = args.data_root / dataset_name / str(setup) / "train.json"
            if not train_json.exists():
                print(f"  [missing] {train_json}")
                continue

            with train_json.open("r", encoding="utf-8") as fh:
                data_dict = json.load(fh)

            # data_dict: {ckpt_key: [{image_path, common, ...}, ...], ...}
            for ckpt, samples in data_dict.items():
                counts = Counter(
                    img[args.label_type] for img in samples
                    if args.label_type in img
                )
                gini = gini_coefficient(list(counts.values()))
                rows.append({
                    "dataset": dataset_name,
                    "setup": setup,
                    "ckpt": ckpt,
                    "num_classes": len(counts),
                    "num_samples": sum(counts.values()),
                    "gini_index": round(gini, 6),
                })
                print(
                    f"{dataset_name} | setup={setup} | ckpt={ckpt} | "
                    f"classes={len(counts)} | samples={sum(counts.values())} | "
                    f"gini={gini:.4f}"
                )

                if args.plot:
                    plot_title = f"{dataset_name} — setup {setup} — {ckpt}"
                    safe_name = dataset_name.replace("/", "_")
                    out_png = args.output_dir / f"{safe_name}_setup{setup}_{ckpt}.png"
                    plot_class_counts(dict(counts), plot_title, out_png)

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "gini_index_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
