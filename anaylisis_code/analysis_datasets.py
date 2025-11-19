#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count images per class for each ckp in train.json and test.json across datasets.

Outputs:
  - Per-dataset CSVs (train_counts.csv, test_counts.csv)
  - One global long-format CSV (all_datasets_counts_long.csv)

No CLI input. Edit CONFIG below.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ============== CONFIG (edit here) ==============
CONFIG = {
    "datasets": {
        # name -> dict of split -> path
        # "serengeti_E08": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/serengeti/serengeti_E08/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/serengeti/serengeti_E08/30/test.json",
        # },
        # "nz_PS1_CAM7709": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_PS1_CAM7709/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_PS1_CAM7709/30/test.json",
        # },
        # "nz_EFH_HCAMH03": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAMH03/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAMH03/30/test.json",
        # },
        # "na_lebec_CA-36": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/na/na_lebec_CA-36/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/na/na_lebec_CA-36/30/test.json",
        # },
        # "nz_HLO_Moto2": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_HLO_Moto2/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_HLO_Moto2/30/test.json",
        # },
        # "nz_EFD_DCAME02": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFD_DCAME02/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFD_DCAME02/30/test.json",
        # },
        # "nz_EFD_DCAMB02": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFD_DCAMB02/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFD_DCAMB02/30/test.json",
        # },
        # "caltech_88": {
        #     "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/caltech/caltech_88/30/train.json",
        #     "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/caltech/caltech_88/30/test.json",
        # },
        "APN_13U":{
            "train": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare/APN/APN_13U/30/train.json",
            "test":  "/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare/APN/APN_13U/30/test.json",
            "rare": "/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare/APN/APN_13U/30/rare.json",
        }
    },
    # Where CSVs are written
    "output_dir": "/users/PAS2099/lemengwang/Documents/icicle/Camera-Trap-CVPR/class_counts",
    # Which fields might contain the class label in each record
    "class_keys": ["common"],
}
# ================================================


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def extract_class(rec: Dict[str, Any], class_keys: List[str]) -> str:
    for k in class_keys:
        if k in rec and isinstance(rec[k], (str, int)):
            return str(rec[k])
    return "unknown"


def count_by_ckp(json_path: Path, class_keys: List[str]) -> Dict[int, Counter]:
    """
    Returns {ckp_index: Counter({class: count, ...}), ...}
    Expects JSON structure with keys like "ckp_1", "ckp_2", ... each being a list of records.
    """
    if not json_path.exists():
        return {}

    data = load_json(json_path)
    counts: Dict[int, Counter] = {}
    for k, v in data.items():
        if isinstance(k, str) and k.startswith("ckp_") and isinstance(v, list):
            try:
                idx = int(k.split("_")[1])
            except Exception:
                continue
            c = Counter()
            for rec in v:
                cls = extract_class(rec if isinstance(rec, dict) else {}, class_keys)
                c[cls] += 1
            counts[idx] = c
    return counts


def write_long_csv(rows: List[Tuple[str, str, int, str, int]], out_path: Path):
    """
    rows: list of (dataset, split, ckp, class, count)
    """
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "split", "ckp", "class", "count"])
        writer.writerows(rows)


def write_wide_csv(counts: Dict[int, Counter], out_path: Path):
    """
    counts: {ckp: Counter(class->count)}
    Writes a wide CSV: rows=ckp, cols=classes, values=count. Missing=0.
    """
    ensure_dir(out_path.parent)
    # Collect full class set
    classes = set()
    for c in counts.values():
        classes.update(c.keys())
    classes = sorted(classes)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ckp"] + classes + ["TOTAL"])
        for ckp in sorted(counts.keys()):
            c = counts[ckp]
            row = [ckp] + [c.get(cls, 0) for cls in classes]
            row.append(sum(c.values()))
            writer.writerow(row)


def main():
    out_root = Path(CONFIG["output_dir"])
    class_keys = CONFIG["class_keys"]

    global_rows: List[Tuple[str, str, int, str, int]] = []

    for dataset_name, splits in CONFIG["datasets"].items():
        ds_out_dir = out_root / dataset_name
        ensure_dir(ds_out_dir)

        for split in ("train", "test", "rare"):
            json_path = Path(splits.get(split, ""))
            counts_by_ckp = count_by_ckp(json_path, class_keys)

            # Per-dataset per-split wide CSV
            wide_csv_path = ds_out_dir / f"{split}_counts.csv"
            write_wide_csv(counts_by_ckp, wide_csv_path)

            # Collect for global long CSV
            for ckp, counter in counts_by_ckp.items():
                for cls, cnt in counter.items():
                    global_rows.append((dataset_name, split, ckp, cls, cnt))

    # Global long CSV
    write_long_csv(global_rows, out_root / "all_datasets_counts_long.csv")

    # Small console summary
    print("✅ Done.")
    print(f"Per-dataset CSVs and global CSV saved under: {out_root}")
    if global_rows:
        # quick preview: top 10 rows
        print("\nPreview (first 10 rows of global long CSV):")
        for r in global_rows[:10]:
            print(r)


if __name__ == "__main__":
    main()
