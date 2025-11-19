#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classify rare images into Type A / B / C for multiple datasets.

Type A (Previously-seen rare):
    Class appears in train ckps < T
Type B (Future-seen rare):
    Class appears in train ckps > T (but not < T)
Type C (True rare, never-seen):
    Class never appears in ANY train ckp

For each dataset and each checkpoint T, we compute:
    - # rare images of type A, B, C
    - %A, %B, %C among rare images at T
    - %A, %B, %C among ALL test images at T (test + rare)

We also compute dataset-level totals (across all ckps):
    - # rare images of type A, B, C
    - %A, %B, %C among all rare images
    - %A, %B, %C among all test images (test + rare)

Outputs:
    One CSV per dataset under OUTPUT_DIR with per-ckp + dataset summary.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# ----------- CONFIG -----------

BASE_DIR = Path("/fs/scratch/PAS2099/camera-trap-benchmark/dataset_rare")
JSON_SUBDIR = "30"  # folder containing train/test/rare.json
OUTPUT_DIR = Path("/fs/ess/PAS2099/camera-trap-CVPR-logs/rare_type_stats")  # where to save CSVs

DATASETS = [
    "serengeti/serengeti_H03",
    "serengeti/serengeti_E05",
    "APN/APN_13U",
    "serengeti/serengeti_L10",
    "serengeti/serengeti_D02",
    "serengeti/serengeti_T10",
    "serengeti/serengeti_Q07",
    "serengeti/serengeti_S11",
    "ENO/ENO_C04",
    "MTZ/MTZ_D03",
]

# ----------- HELPERS -----------

def load_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load JSON file if it exists, else return empty dict."""
    if not path.is_file():
        print(f"[WARN] Missing file: {path}")
        return {}
    with path.open("r") as f:
        return json.load(f)


def build_train_presence(train_data: Dict[str, List[Dict[str, Any]]]) -> Dict[int, List[int]]:
    """
    Build a map:
        class_id -> sorted list of ckps where it appears in train.
    """
    presence = defaultdict(set)  # class_id -> set of ckp indices

    for ckp_name, entries in train_data.items():
        # assume keys like "ckp_1", "ckp_2", ...
        try:
            t = int(ckp_name.split("_")[1])
        except (IndexError, ValueError):
            print(f"[WARN] Unexpected ckp key format: {ckp_name}")
            continue

        for e in entries:
            cid = int(e["class_id"])
            presence[cid].add(t)

    # convert sets to sorted lists
    return {cid: sorted(list(ts)) for cid, ts in presence.items()}


def classify_rare_type(
    class_id: int,
    t: int,
    train_presence: Dict[int, List[int]],
) -> str:
    """
    Classify a rare sample at checkpoint t with class_id into A/B/C.

    Type A: seen in train ckps < t
    Type B: not A, but seen in train ckps > t
    Type C: not seen in any train ckps
    """
    ts = train_presence.get(class_id, [])

    seen_before = any(tt < t for tt in ts)
    seen_after = any(tt > t for tt in ts)

    if seen_before:
        return "A"
    elif seen_after:
        return "B"
    else:
        return "C"


def safe_pct(num: int, denom: int) -> float:
    """Return num/denom as float, or 0.0 if denom == 0."""
    return float(num) / float(denom) if denom > 0 else 0.0


# ----------- MAIN ANALYSIS -----------

def analyze_dataset(dataset: str) -> None:
    """
    For a single dataset (e.g. 'APN/APN_13U'), compute per-ckp and
    dataset-level A/B/C percentages and write a CSV.
    """
    json_dir = BASE_DIR / dataset / JSON_SUBDIR
    train_path = json_dir / "train.json"
    test_path = json_dir / "test.json"
    rare_path = json_dir / "rare.json"

    print(f"\n=== Dataset: {dataset} ===")
    print(f"Loading: {train_path}, {test_path}, {rare_path}")

    train_data = load_json(train_path)
    test_data = load_json(test_path)
    rare_data = load_json(rare_path)

    # Build train presence map
    train_presence = build_train_presence(train_data)

    # Collect all checkpoint names from union of train/test/rare keys
    all_ckps = set()
    all_ckps.update(train_data.keys())
    all_ckps.update(test_data.keys())
    all_ckps.update(rare_data.keys())

    # Filter to ckp_* pattern and sort by index
    def ckp_key_to_t(ckp_name: str) -> int:
        try:
            return int(ckp_name.split("_")[1])
        except Exception:
            return 10**9  # push weird keys to the end

    sorted_ckps = sorted(all_ckps, key=ckp_key_to_t)

    # Aggregates for dataset-level totals
    total_rare_A = total_rare_B = total_rare_C = 0
    total_rare_all = 0
    total_test_all = 0  # test + rare

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = dataset.replace("/", "_") + "_rare_type_stats.csv"
    out_path = OUTPUT_DIR / out_name

    fieldnames = [
        "dataset",
        "ckp",
        "ckp_index",
        "rare_count",
        "rare_A",
        "rare_B",
        "rare_C",
        "rare_pct_A",
        "rare_pct_B",
        "rare_pct_C",
        "test_count_without_rare",
        "test_total_with_rare",
        "test_pct_A",
        "test_pct_B",
        "test_pct_C",
    ]

    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ckp_name in sorted_ckps:
            # ignore non-ckp-like keys if any
            if not ckp_name.startswith("ckp_"):
                continue

            t = ckp_key_to_t(ckp_name)

            rare_entries = rare_data.get(ckp_name, [])
            test_entries = test_data.get(ckp_name, [])

            rare_A = rare_B = rare_C = 0

            # classify each rare image
            for e in rare_entries:
                cid = int(e["class_id"])
                rtype = classify_rare_type(cid, t, train_presence)
                if rtype == "A":
                    rare_A += 1
                elif rtype == "B":
                    rare_B += 1
                else:
                    rare_C += 1

            rare_count = len(rare_entries)
            test_without_rare = len(test_entries)
            test_total = test_without_rare + rare_count

            # percentages within rare subset
            rare_pct_A = safe_pct(rare_A, rare_count)
            rare_pct_B = safe_pct(rare_B, rare_count)
            rare_pct_C = safe_pct(rare_C, rare_count)

            # percentages among all test images (test + rare)
            test_pct_A = safe_pct(rare_A, test_total)
            test_pct_B = safe_pct(rare_B, test_total)
            test_pct_C = safe_pct(rare_C, test_total)

            # update dataset-level totals
            total_rare_A += rare_A
            total_rare_B += rare_B
            total_rare_C += rare_C
            total_rare_all += rare_count
            total_test_all += test_total

            writer.writerow(
                dict(
                    dataset=dataset,
                    ckp=ckp_name,
                    ckp_index=t,
                    rare_count=rare_count,
                    rare_A=rare_A,
                    rare_B=rare_B,
                    rare_C=rare_C,
                    rare_pct_A=rare_pct_A,
                    rare_pct_B=rare_pct_B,
                    rare_pct_C=rare_pct_C,
                    test_count_without_rare=test_without_rare,
                    test_total_with_rare=test_total,
                    test_pct_A=test_pct_A,
                    test_pct_B=test_pct_B,
                    test_pct_C=test_pct_C,
                )
            )

        # ---- Dataset-level summary row ----
        dataset_rare_pct_A = safe_pct(total_rare_A, total_rare_all)
        dataset_rare_pct_B = safe_pct(total_rare_B, total_rare_all)
        dataset_rare_pct_C = safe_pct(total_rare_C, total_rare_all)

        dataset_test_pct_A = safe_pct(total_rare_A, total_test_all)
        dataset_test_pct_B = safe_pct(total_rare_B, total_test_all)
        dataset_test_pct_C = safe_pct(total_rare_C, total_test_all)

        writer.writerow(
            dict(
                dataset=dataset,
                ckp="ALL",
                ckp_index=-1,
                rare_count=total_rare_all,
                rare_A=total_rare_A,
                rare_B=total_rare_B,
                rare_C=total_rare_C,
                rare_pct_A=dataset_rare_pct_A,
                rare_pct_B=dataset_rare_pct_B,
                rare_pct_C=dataset_rare_pct_C,
                test_count_without_rare="",  # not meaningful at ALL level
                test_total_with_rare=total_test_all,
                test_pct_A=dataset_test_pct_A,
                test_pct_B=dataset_test_pct_B,
                test_pct_C=dataset_test_pct_C,
            )
        )

    print(f"[OK] Wrote {out_path}")


def main():
    for ds in DATASETS:
        analyze_dataset(ds)


if __name__ == "__main__":
    main()
