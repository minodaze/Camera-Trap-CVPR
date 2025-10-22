import os
import json
import pandas as pd
from typing import Optional

CSV_IN = "/users/PAS2099/mino/ICICLE/other/CL + Animal Trap - Camera-Ready (50).csv"
CSV_OUT = "/users/PAS2099/mino/ICICLE/other/CL + Animal Trap - Camera-Ready-filled (50).csv"

DATA_ROOT = "/fs/scratch/PAS2099/camera-trap-final/camera_ready/"
# Examples:
#   camera_ready/APN_APN_13U/best_accum/bioclip2/lora_8_text_head/all/log/final_training_summary.json
#   camera_ready/APN_APN_13U/base_accum/bioclip2/full_text_head/all/log/final_training_summary.json


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[WARN] Malformed JSON {path}: {e}")
        return None


def main():
    df = pd.read_csv(CSV_IN)

    # Ensure output columns exist
    if 'BASE Accum.' not in df.columns:
        df['BASE Accum.'] = pd.NA
    if 'BEST Accum.' not in df.columns:
        df['BEST Accum.'] = pd.NA

    for i, row in df.iterrows():
        dataset_name = row['dataset']
        dn = str(dataset_name).replace("/", "_")
        data_dir = os.path.join(DATA_ROOT, dn)

        best_path = os.path.join(
            data_dir, "best_accum/bioclip2/lora_8_text_head/all/log/final_training_summary.json"
        )
        base_path = os.path.join(
            data_dir, "base_accum/bioclip2/full_text_head/all/log/final_training_summary.json"
        )

        best_data = read_json(best_path)
        base_data = read_json(base_path)

        best_balanced_acc = (
            best_data.get('averages', {}).get('balanced_accuracy') if best_data else pd.NA
        )
        base_balanced_acc = (
            base_data.get('averages', {}).get('balanced_accuracy') if base_data else pd.NA
        )

        # IMPORTANT: write back to the DataFrame using .loc/.at (iterrows returns a copy)
        df.loc[i, 'BASE Accum.'] = base_balanced_acc
        df.loc[i, 'BEST Accum.'] = best_balanced_acc

    df.to_csv(CSV_OUT, index=False)
    print(f"[OK] Wrote filled CSV to: {CSV_OUT}")


if __name__ == "__main__":
    main()