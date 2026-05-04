import os
import pandas as pd
import json

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - eccv 2 (1).csv')

thershold = 6
large_interval_datasets = []
small_interval_datasets = []
for dataset in df['dataset'].to_list():
    dataset_ = dataset.replace('/', '_')
    accum_eval_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_accu_eval_all/{dataset_}/bioclip2/lora_8_text_head/all/log/eval_accu_eval_only_summary.json'
    if not os.path.exists(accum_eval_path):
        accum_eval_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_accu_eval/{dataset_}/bioclip2/lora_8_text_head/all/log/eval_accu_eval_only_summary.json'

    with open(accum_eval_path, 'r') as f:
        accum_eval = json.load(f)

    if df.loc[df['dataset'] == dataset, 'zs'].values[0] < 0.8:
        if len(accum_eval) - 1 > thershold:
            large_interval_datasets.append(dataset_)
        else:
            small_interval_datasets.append(dataset_)

print(f"Datasets with large intervals between checkpoints: {len(large_interval_datasets)}")
print(f"Datasets with small intervals between checkpoints: {len(small_interval_datasets)}")

large_ckpt_ratio = [0, 0.25, 0.5, 0.75, 1.0]
small_ckpt_ratio = [0, 0.5, 1.0]

large_interval_datasets_avg_stats = {0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}
small_interval_datasets_avg_stats = {0: [], 0.5: [], 1.0: []}

def get_avg_ba(accum_eval: dict, ckpt_key: int):
    avg_ba = 0
    num_ckpts = 0
    for key, value in accum_eval.items():
        ckpt_num = int(key.split('_')[1]) if key.startswith('ckp_') else None
        if ckpt_num >= ckpt_key and key.startswith('ckp_') and isinstance(value, dict) and 'balanced_accuracy' in value:
            avg_ba += value['balanced_accuracy']
            num_ckpts += 1
    return avg_ba / num_ckpts if num_ckpts > 0 else 0

print("Datasets with large intervals between checkpoints:")
for ds in large_interval_datasets:
    print(f"  - {ds}")
    accum_eval_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_accu_eval_all/{ds}/bioclip2/lora_8_text_head/all/log/eval_accu_eval_only_summary.json'
    if not os.path.exists(accum_eval_path):
        accum_eval_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_accu_eval/{ds}/bioclip2/lora_8_text_head/all/log/eval_accu_eval_only_summary.json'

    with open(accum_eval_path, 'r') as f:
        accum_eval = json.load(f)
    
    total_ckpts = len(accum_eval) - 1
    print(f"  - Number of checkpoints: {total_ckpts}")
    for ratio in large_ckpt_ratio:
        idx = int(ratio * (total_ckpts - 1))
        ckpt_key = f'ckp_{idx+1}'
        if ckpt_key in accum_eval:
            avg_ba = get_avg_ba(accum_eval[ckpt_key], idx+1)
            print(f"    - Checkpoint {ckpt_key} (ratio {ratio:.2f}): balanced_accuracy = {avg_ba}")
        else:
            print(f"    - Checkpoint {ckpt_key} (ratio {ratio:.2f}): not found")
        large_interval_datasets_avg_stats[ratio].append(avg_ba)

print("\nDatasets with small intervals between checkpoints:")
for ds in small_interval_datasets:
    print(f"  - {ds}")
    accum_eval_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_accu_eval_all/{ds}/bioclip2/lora_8_text_head/all/log/eval_accu_eval_only_summary.json'
    if not os.path.exists(accum_eval_path):
        accum_eval_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_accu_eval/{ds}/bioclip2/lora_8_text_head/all/log/eval_accu_eval_only_summary.json'
    
    with open(accum_eval_path, 'r') as f:
        accum_eval = json.load(f)
    total_ckpts = len(accum_eval) - 1
    print(f"  - Number of checkpoints: {total_ckpts}")
    for ratio in small_ckpt_ratio:
        idx = int(ratio * (total_ckpts - 1))
        ckpt_key = f'ckp_{idx+1}'
        if ckpt_key in accum_eval:
            avg_ba = get_avg_ba(accum_eval[ckpt_key], idx+1)
            print(f"    - Checkpoint {ckpt_key} (ratio {ratio:.2f}): balanced_accuracy = {avg_ba}")
        else:
            print(f"    - Checkpoint {ckpt_key} (ratio {ratio:.2f}): not found")
        small_interval_datasets_avg_stats[ratio].append(avg_ba)

print("\nAverage balanced_accuracy for large interval datasets:")
print(f"  - Total datasets: {len(large_interval_datasets)}")
for ratio, stats in large_interval_datasets_avg_stats.items():
    avg_stat = sum(stats) / len(stats) if stats else 0
    print(f"  - Ratio {ratio:.2f}: average balanced_accuracy = {avg_stat}")

print("\nAverage balanced_accuracy for small interval datasets:")
print(f"  - Total datasets: {len(small_interval_datasets)}")
for ratio, stats in small_interval_datasets_avg_stats.items():
    avg_stat = sum(stats) / len(stats) if stats else 0
    print(f"  - Ratio {ratio:.2f}: average balanced_accuracy = {avg_stat}")