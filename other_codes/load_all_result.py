import os
import pandas as pd
import json

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/final_all_cameras_percentage - final_all_cameras_percentage.csv')
df['Best Accum'] = 0.0
df['Accum'] = 0.0

with open('/users/PAS2099/mino/ICICLE/uselist/all_cam.txt', 'r') as f:
    datasets = [line.strip() for line in f.readlines()]

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
    return total / weight_sum if weight_sum > 0 else 0.0

# best_accum_updated = 0
missing_rows = 0

# for dataset in datasets:
#     dataset_raw = dataset.strip()
#     dataset_slug = dataset_raw.replace('/', '_')
#     json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum/{dataset_slug}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
#     # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
#     if not os.path.isfile(json_path):
#         json_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/{dataset_slug}/bioclip2/lora_8_text_head/all/log/final_training_summary.json'
#         # /fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/APN_APN_U23A/bioclip2/lora_8_text_head/all/log/final_training_summary.json
#         if not os.path.isfile(json_path):
#             json_path = f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_accum/{dataset_slug}/final_training_summary.json'
#             if not os.path.isfile(json_path):
#                 # print(f"[WARN] JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
#                 continue
#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#         ba = extract_balanced_accuracy(data)
#         mask = df['dataset'] == dataset_raw
#         if not bool(mask.any()):
#             missing_rows += 1
#             print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
#             continue
#         df.loc[mask, 'Best Accum'] = ba
#         best_accum_updated += int(mask.sum())
#     except Exception as e:
#         print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
#         continue

# print(f"Updated rows for accum: {best_accum_updated} (missing CSV rows: {missing_rows})")

# accum_updated = 0

# for dataset in datasets:
#     dataset_raw = dataset.strip()
#     dataset_slug = dataset_raw.replace('/', '_')
#     json_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/accum/{dataset_slug}/bioclip2/full_text_head/all/log/final_training_summary.json'
#     # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
#     if not os.path.isfile(json_path):
#         json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/accum/{dataset_slug}/bioclip2/full_text_head_loss_ce/all/log/final_training_summary.json'
#         # /fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/APN_APN_U23A/bioclip2/lora_8_text_head/all/log/final_training_summary.json
#         if not os.path.isfile(json_path):
#             # print(f"[WARN] JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
#             continue
#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#         ba = extract_balanced_accuracy(data)
#         mask = df['dataset'] == dataset_raw
#         if not bool(mask.any()):
#             missing_rows += 1
#             print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
#             continue
#         df.loc[mask, 'Accum'] = ba
#         accum_updated += int(mask.sum())
#     except Exception as e:
#         print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
#         continue
# print(f"Updated rows for accum: {accum_updated} (missing CSV rows: {missing_rows})")

best_oracle_updated = 0

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    json_path = f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle/{dataset_slug}/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    if not os.path.isfile(json_path):
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_raw
        if not bool(mask.any()):
            missing_rows += 1
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, 'best oracle'] = ba
        best_oracle_updated += int(mask.sum())
    except Exception as e:
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

out_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-all results(new).csv'
df.to_csv(out_path, index=False)
print(f"Updated rows for best oracle: {best_oracle_updated} (missing CSV rows: {missing_rows})")
print(f"Updated CSV saved to {out_path}")