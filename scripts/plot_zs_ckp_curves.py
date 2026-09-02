import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATASET_LIST = Path("/users/PAS2099/mino/ICICLE/uselist/15_60.txt")
DEFAULT_OUTPUT_DIR = Path("/users/PAS2099/mino/ICICLE/plots/zs_ckp_curves")
DEFAULT_ROOTS = {
    "zs": Path("/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/zs_30"),
    "zs_15": Path("/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/zs_15"),
    "zs_60": Path("/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/zs_60"),
}
JSON_RELATIVE_PATH = Path("bioclip2/full_text_head_loss_ce/log/final_training_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot checkpoint-wise zero-shot curves for zs, zs_15, and zs_60 from "
            "final_training_summary.json files."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Dataset names like caltech/caltech_88 or already-normalized slugs like "
            "caltech_caltech_88. Defaults to all datasets in --dataset-list."
        ),
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        default=DEFAULT_DATASET_LIST,
        help="Text file with one dataset per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG plots and the summary CSV will be written.",
    )
    parser.add_argument(
        "--zs-root",
        type=Path,
        default=DEFAULT_ROOTS["zs"],
        help="Root directory for baseline zs summaries.",
    )
    parser.add_argument(
        "--zs15-root",
        type=Path,
        default=DEFAULT_ROOTS["zs_15"],
        help="Root directory for zs_15 summaries.",
    )
    parser.add_argument(
        "--zs60-root",
        type=Path,
        default=DEFAULT_ROOTS["zs_60"],
        help="Root directory for zs_60 summaries.",
    )
    parser.add_argument(
        "--metric",
        default="balanced_accuracy",
        help="Metric key to plot from each checkpoint result.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open each plot interactively after saving it.",
    )
    return parser.parse_args()


def normalize_dataset(dataset: str) -> str:
    return dataset.strip().replace("/", "_")


def load_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return [dataset.strip() for dataset in args.datasets if dataset.strip()]

    with args.dataset_list.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def summary_path(root: Path, dataset_slug: str) -> Path:
    return root / dataset_slug / JSON_RELATIVE_PATH


def extract_checkpoint_metric(summary: dict, metric_name: str) -> pd.Series:
    checkpoint_results = summary.get("checkpoint_results")
    if not isinstance(checkpoint_results, dict) or not checkpoint_results:
        checkpoint_results = {
            key: value
            for key, value in summary.items()
            if isinstance(key, str) and key.startswith("ckp_") and isinstance(value, dict)
        }

    if not checkpoint_results:
        raise KeyError("No checkpoint results found in summary JSON")

    rows = []
    for checkpoint_name, checkpoint_metrics in checkpoint_results.items():
        if metric_name not in checkpoint_metrics:
            raise KeyError(f"Metric '{metric_name}' missing in {checkpoint_name}")
        checkpoint_index = int(str(checkpoint_name).split("_")[-1])
        rows.append((checkpoint_index, float(checkpoint_metrics[metric_name])))

    rows.sort(key=lambda item: item[0])
    return pd.Series(
        data=[value for _, value in rows],
        index=[checkpoint_index for checkpoint_index, _ in rows],
        dtype=float,
    )


def load_curve(json_path: Path, metric_name: str) -> pd.Series:
    with json_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return extract_checkpoint_metric(summary, metric_name)


def plot_dataset_curves(
    dataset_name: str,
    curves: dict[str, pd.Series],
    metric_name: str,
    output_path: Path,
    show: bool,
) -> None:
    plt.figure(figsize=(10, 6))
    for label, series in curves.items():
        plt.plot(series.index, series.values, marker="o", linewidth=2, label=label)

    plt.xlabel("Checkpoint")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"Checkpoint-wise ZS curves: {dataset_name}")
    plt.xticks(sorted({int(index) for series in curves.values() for index in series.index}))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close()


def main() -> None:
    args = parse_args()
    datasets = load_datasets(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        "zs": args.zs_root,
        "zs_15": args.zs15_root,
        "zs_60": args.zs60_root,
    }

    all_rows = []
    missing_rows = []

    for dataset_name in datasets:
        dataset_slug = normalize_dataset(dataset_name)
        curves = {}

        for label, root in roots.items():
            json_path = summary_path(root, dataset_slug)
            if not json_path.is_file():
                missing_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_slug": dataset_slug,
                        "curve": label,
                        "json_path": str(json_path),
                    }
                )
                print(f"[WARN] Missing {label} summary: {json_path}")
                continue

            try:
                curve = load_curve(json_path, args.metric)
            except Exception as exc:
                missing_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_slug": dataset_slug,
                        "curve": label,
                        "json_path": str(json_path),
                        "error": str(exc),
                    }
                )
                print(f"[ERROR] Failed to load {label} for {dataset_name}: {exc}")
                continue

            curves[label] = curve
            for checkpoint, value in curve.items():
                all_rows.append(
                    {
                        "dataset": dataset_name,
                        "curve": label,
                        "checkpoint": int(checkpoint),
                        args.metric: value,
                    }
                )

        if not curves:
            print(f"[WARN] No curves found for dataset {dataset_name}")
            continue

        output_path = output_dir / f"{dataset_slug}_zs_ckp_curves.png"
        plot_dataset_curves(dataset_name, curves, args.metric, output_path, args.show)
        print(f"[INFO] Saved plot: {output_path}")

    curves_csv_path = output_dir / "zs_ckp_curves.csv"
    pd.DataFrame(all_rows).to_csv(curves_csv_path, index=False)
    print(f"[INFO] Saved curve table: {curves_csv_path}")

    if missing_rows:
        missing_csv_path = output_dir / "zs_ckp_missing.csv"
        pd.DataFrame(missing_rows).to_csv(missing_csv_path, index=False)
        print(f"[INFO] Saved missing summary table: {missing_csv_path}")


if __name__ == "__main__":
    main()