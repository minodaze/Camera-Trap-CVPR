import os
import pandas as pd
import json

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/Camera-Trap ECCV Rebuttal - new_15_16.csv')

unfinished_accum_15 = []
unfinished_accum_60 = []
unfinished_best_accum_15 = []
unfinished_best_accum_60 = []
unfinished_best_oracle_15 = []
unfinished_best_oracle_60 = []

with open('/users/PAS2099/mino/ICICLE/uselist/15_60.txt', 'r') as f:
    datasets = [line.strip() for line in f.readlines()]
print(df.head())

accum_15_updated = 0
accum_60_updated = 0
best_accum_15_updated = 0
best_accum_60_updated = 0
er_lora_bsm_updated = 0
seq_updated = 0
zs_15_updated = 0
zs_60_updated = 0

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

best_accum_updated = 0

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/accum_15_bio/{dataset_slug}/bioclip2/full_text_head_loss_ce/all/log/final_training_summary.json'
    if not os.path.isfile(json_path):
        unfinished_accum_15.append(dataset_raw)
        print(f"[WARN] accum_15 JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_slug}")
            continue
        df.loc[mask, '15-accum-bio'] = ba
        accum_15_updated += int(mask.sum())
    except Exception as e:
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for 15-accum-bio: {accum_15_updated})")

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/accum_60_bio/{dataset_slug}/bioclip2/full_text_head_loss_ce/all/log/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/accum_60_bio/APN_APN_K082/bioclip2/full_text_head_loss_ce/all/log/final_training_summary.json
    if not os.path.isfile(json_path):
        unfinished_accum_60.append(dataset_raw)
        print(f"[WARN] accum_60 JSON not found for dataset {dataset_raw} ({json_path})")
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, '60-accum-bio'] = ba
        accum_60_updated += int(mask.sum())
    except Exception as e:
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for 60-accum-bio: {accum_60_updated})")

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_60_bio/{dataset_slug}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_openai/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    
    if not os.path.isfile(json_path):
        print(f"[WARN] 60-best accum-bio JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
        unfinished_best_accum_60.append(dataset_raw)
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, '60-best accum-bio'] = ba
        best_accum_60_updated += int(mask.sum())
    except Exception as e:
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for 60-best accum-bio: {best_accum_60_updated})")

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_15_bio/nz_nz_EFH_HCAME08/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_15_bio/{dataset_slug}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_bio/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    
    if not os.path.isfile(json_path):
        print(f"[WARN] 15-best accum-bio JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
        unfinished_best_accum_15.append(dataset_raw)
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, '15-best accum-bio'] = ba
        best_accum_15_updated += int(mask.sum())
    except Exception as e:
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for 15-best accum-bio: {best_accum_15_updated})")

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/zs_60_bioclip/{dataset_slug}/bioclip2/full_text_head_loss_ce/log/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_openai/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    
    if not os.path.isfile(json_path):
        print(f"[WARN] 60-zs-openai JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, '60-zs-openai'] = ba
        zs_60_updated += int(mask.sum())
    except Exception as e:
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for 60-zs-openai: {zs_60_updated})")

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_15_openai/nz_nz_EFH_HCAME08/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/zs_15_bioclip/{dataset_slug}/bioclip2/full_text_head_loss_ce/log/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_openai/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    if not os.path.isfile(json_path):
        print(f"[WARN] 15-zs-openai JSON not found for dataset {dataset_raw} ({json_path})")
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, '15-zs-openai'] = ba
        zs_15_updated += int(mask.sum())
    except Exception as e:
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for 15-zs-openai: {zs_15_updated})")

seq_lora_bsm_updated = 0

unfinished_seq = []
unfinished_er = []

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_seq_openai_new/{dataset_slug}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_openai/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    
    if not os.path.isfile(json_path):
        print(f"[WARN] Seq + LoRA & BSM JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, 'Seq + LoRA & BSM'] = ba
        seq_lora_bsm_updated += int(mask.sum())
    except Exception as e:
        unfinished_seq.append(dataset_raw)
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for Seq + LoRA & BSM: {seq_lora_bsm_updated})")

er_lora_bsm_updated = 0

for dataset in datasets:
    dataset_raw = dataset.strip()
    dataset_slug = dataset_raw.replace('/', '_')
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_er2_new/{dataset_slug}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_accum_openai/CDB_CDB_E02/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json
    
    if not os.path.isfile(json_path):
        print(f"[WARN] ER + LoRA & BSM JSON not found for dataset {dataset_raw} (slug: {dataset_slug})")
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ba = extract_balanced_accuracy(data)
        mask = df['dataset'] == dataset_slug
        if not bool(mask.any()):
            print(f"[WARN] Dataset not found in CSV: {dataset_raw}")
            continue
        df.loc[mask, 'ER + LoRA & BSM'] = ba
        er_lora_bsm_updated += int(mask.sum())
    except Exception as e:
        unfinished_er.append(dataset_raw)
        print(f"[ERROR] Failed to read JSON for dataset {dataset_raw}: {e}")
        continue

print(f"Updated rows for ER + LoRA & BSM: {er_lora_bsm_updated})")

out_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-all results(15_60).csv'

# Skip rows that missed 15/60 results.
before_rows = len(df)
df = df.dropna(subset=['ER + LoRA & BSM']).copy()
after_rows = len(df)
print(f"Dropping rows missing ER + LoRA & BSM: {before_rows - after_rows} (kept {after_rows})")

df.to_csv(out_path, index=False)

print('unfinished_accum_15:')
for dataset in unfinished_accum_15:
    print(f"{dataset}")
print('unfinished_accum_60:')
for dataset in unfinished_accum_60:
    print(f"{dataset}")
print('unfinished_best_accum_15:')
for dataset in unfinished_best_accum_15:
    print(f"{dataset}")
print('unfinished_best_accum_60:')
for dataset in unfinished_best_accum_60:
    print(f"{dataset}")
print('unfinished_er:')
for dataset in unfinished_er:
    print(f"{dataset}")
print('unfinished_seq:')
for dataset in unfinished_seq:
    print(f"{dataset}")