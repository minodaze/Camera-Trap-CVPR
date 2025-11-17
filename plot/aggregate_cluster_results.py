import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

ORACLE_LIST = "/users/PAS2099/mino/ICICLE/plot/oracle.txt"

# Root where logs live
ROOT = "/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20"

# Paths templates for final_training_summary.json per method/cluster
PATHS = {
    "bsm": {
        "ascend": os.path.join(
            ROOT,
            "bsm_oracle_test_per_epoch_ascend",
            "{dataset}",
            "bioclip2",
            "full_text_head",
            "all",
            "log",
            "final_training_summary.json",
        ),
        "cardinal": os.path.join(
            ROOT,
            "bsm_oracle_test_per_epoch_cardinal",
            "{dataset}",
            "bioclip2",
            "full_text_head",
            "all",
            "log",
            "final_training_summary.json",
        ),
    },
    "lora": {
        "ascend": os.path.join(
            ROOT,
            "lora_oracle_test_per_epoch_ascend",
            "{dataset}",
            "bioclip2",
            "lora_8_text_head",
            "all",
            "log",
            "final_training_summary.json",
        ),
        "cardinal": os.path.join(
            ROOT,
            "lora_oracle_test_per_epoch_cardinal",
            "{dataset}",
            "bioclip2",
            "lora_8_text_head",
            "all",
            "log",
            "final_training_summary.json",
        ),
    },
}


def _safe_load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_balanced_accuracy(summary: Dict[str, Any]) -> Optional[float]:
    if not summary:
        return None
    # Primary expected location
    v = (
        summary.get("averages", {}).get("balanced_accuracy")
        if isinstance(summary.get("averages"), dict)
        else None
    )
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    # Fallbacks
    for key in (
        ("test_avg", "avg_balanced_acc"),
        ("averages", "avg_balanced_acc"),
    ):
        outer, inner = key
        outer_obj = summary.get(outer, {}) if isinstance(summary, dict) else {}
        if isinstance(outer_obj, dict):
            v2 = outer_obj.get(inner)
            if v2 is not None:
                try:
                    return float(v2)
                except Exception:
                    continue
    return None


def get_metric(dataset: str, method: str, cluster: str) -> Optional[float]:
    template = PATHS[method][cluster]
    path = template.format(dataset=dataset)
    data = _safe_load(path)
    if data is None:
        # Not found, return None
        return None
    return _extract_balanced_accuracy(data)


def build_dataframe(datasets):
    rows = []
    for ds in datasets:
        row = {
            "dataset": ds,
            "bsm_on_ascend": get_metric(ds, "bsm", "ascend"),
            "bsm_on_cardinal": get_metric(ds, "bsm", "cardinal"),
            "lora_on_ascend": get_metric(ds, "lora", "ascend"),
            "lora_on_cardinal": get_metric(ds, "lora", "cardinal"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    # Keep order and types; no scaling to % here to keep raw numbers consistent with summaries
    return df


def main():
    with open(ORACLE_LIST, "r") as f:
        datasets = [line.strip() for line in f if line.strip()]
    df = build_dataframe(datasets)

    # Output locations
    out_dir = "/users/PAS2099/mino/ICICLE/csv"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "cluster_compare_bsm_lora.csv")

    df.to_csv(out_csv, index=False)
    # Also print a compact preview
    with pd.option_context("display.max_rows", 200, "display.width", 180):
        print(df)
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
