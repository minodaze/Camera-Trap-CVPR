import pandas
import json
import os


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

df = pandas.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - ECCV (FINAL).csv')

with open('/users/PAS2099/mino/ICICLE/other_codes/eecv2(f).txt', 'r') as f:
    datasets = [line.strip() for line in f.readlines()]

ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for dataset in datasets:
    dataset_path = dataset.replace("/", "_")
    max_ba = 0.0
    max_ratio = 0.0
    for ratio in ratios:
        json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_inter/{dataset_path}/bioclip2/lora_8_text_head_loss_bsm_lora_interpolate_{ratio}/log/eval_only_summary.json'
        # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_inter/KGA_KGA_KHOGC05/bioclip2/lora_8_text_head_loss_bsm_lora_interpolate_0.1/log/eval_only_predictions.json
        # import pdb; pdb.set_trace()
        if not os.path.isfile(json_path):
            print(f"[WARN] JSON not found for dataset {dataset} at ratio {ratio}")
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            ba = extract_balanced_accuracy(data)
            if ba > max_ba:
                max_ba = ba
                max_ratio = ratio
        except Exception as e:
            print(f"[ERROR] Failed to read JSON for dataset {dataset} at ratio {ratio}: {e}")
            continue
    print(f"Dataset: {dataset}, Max Balanced Accuracy: {max_ba:.4f} at Ratio: {max_ratio:.1f}")
    # df.loc[df['dataset'] == dataset, 'best accum best inter ratio'] = max_ratio
    print(f"Best ratio for dataset {dataset}: {max_ratio}")
    df.loc[df['dataset'] == dataset, 'Best Interpolation (Hongjie)'] = max_ba

df.to_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - ECCV (loaded FINAL).csv', index=False)