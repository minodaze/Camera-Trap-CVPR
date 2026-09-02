import pandas as pd
import json
import os
import numpy as np


def extract_balanced_accuracy(summary: dict) -> float:
    """Return balanced_accuracy from eval_only_summary.json.

    Supports both historical key variants: 'average' and 'averages'.
    Falls back to averaging checkpoint entries if needed.
    """

    if 'averages' in summary and isinstance(summary['averages'], dict):
        return float(summary['averages']['balanced_accuracy'])
    if 'average' in summary and isinstance(summary['average'], dict):
        return float(summary['average']['balanced_accuracy'])

    # Fallback: compute from ckp_* entries if present.
    ckps = [v for k, v in summary.items() if isinstance(k, str) and k.startswith('ckp_') and isinstance(v, dict)]
    if not ckps:
        raise KeyError("Missing 'averages'/'average' and no 'ckp_*' entries")

    # Prefer a sample-weighted mean if num_samples exists.
    total = 0.0
    weight_sum = 0.0
    for c in ckps:
        ba = float(c['balanced_accuracy'])
        w = float(c.get('num_samples', 1.0))
        total += ba * w
        weight_sum += w
    return total / weight_sum if weight_sum else float('nan')

with open('/users/PAS2099/mino/ICICLE/other_codes/eecv2(f).txt', 'r') as f:
    datasets = [line.strip() for line in f.readlines()]

ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratio_cols = [f'BA_ratio_{r:.1f}' for r in ratios]

rows = []
for dataset in datasets:
    dataset_path = dataset.replace("/", "_")
    row = {'dataset': dataset}
    best_ba = 0.0
    best_ratio = float('nan')

    for ratio, col in zip(ratios, ratio_cols):
        json_path = (
            f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_inter/'
            f'{dataset_path}/bioclip2/lora_8_text_head_loss_bsm_lora_interpolate_{ratio}'
            f'/log/eval_only_summary.json'
        )
        if not os.path.isfile(json_path):
            print(f"[WARN] JSON not found for dataset {dataset} at ratio {ratio}")
            row[col] = float('nan')
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            ba = extract_balanced_accuracy(data)
            row[col] = ba
            if ba > best_ba:
                best_ba = ba
                best_ratio = ratio
        except Exception as e:
            print(f"[ERROR] Dataset {dataset} ratio {ratio}: {e}")
            row[col] = float('nan')

    row['Best Interpolation'] = best_ba
    row['Best Ratio'] = best_ratio
    rows.append(row)
    print(f"{dataset}: best BA={best_ba:.4f} at ratio={best_ratio:.1f}")

df = pd.DataFrame(rows, columns=['dataset'] + ratio_cols + ['Best Interpolation', 'Best Ratio'])

# Last row: average improvement of each ratio vs no-interpolation (ratio=1.0)
no_interp_col = 'BA_ratio_1.0'
avg_row = {'dataset': 'Avg improvement vs ratio=1.0'}
for col in ratio_cols:
    diff = df[col] - df[no_interp_col]
    avg_row[col] = round(float(diff.mean(skipna=True)), 4)
avg_row['Best Interpolation'] = round(float((df['Best Interpolation'] - df[no_interp_col]).mean(skipna=True)), 4)
avg_row['Best Ratio'] = ''

df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

out_path = '/users/PAS2099/mino/ICICLE/csv/best_accum_inter_results.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(df.to_string(index=False))