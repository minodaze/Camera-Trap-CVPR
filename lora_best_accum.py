import pandas as pd
import os
import json

# NOTE: Original code read from 'other/' and wrote to 'csv/'. That guarantees the file you inspect in 'csv/' never updates.
# Align read & write to the same CSV path.
CSV_PATH = 'csv/camera-trap-CVPR - (Correct best accum best ratio) Final Best Accum List.csv'
ALT_CSV_PATH = 'other/camera-trap-CVPR - (Correct) Final Best Accum List.csv'
if os.path.isfile(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
elif os.path.isfile(ALT_CSV_PATH):
    df = pd.read_csv(ALT_CSV_PATH)
else:
    raise FileNotFoundError(f"Neither {CSV_PATH} nor {ALT_CSV_PATH} exist")
# /fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/ENO_ENO_E06/bioclip2/lora_8_text_head/all/log
dataset_path = 'eval_dataset.txt'
with open(dataset_path, 'r') as f:
    datasets = [line.strip() for line in f.readlines()]

json_path = []
datasets_norm = []
for ds in datasets:
    # Normalize dataset name to match how directories are stored (slashes -> underscores)
    ds_dir = ds.replace('/', '_')
    datasets_norm.append(ds_dir)
    base_dir = '/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum'
    model_path = f"{base_dir}/{ds_dir}/bioclip2/lora_8_text_head/all/log"
    json_file = os.path.join(model_path, 'final_training_summary.json')
    if not os.path.isfile(json_file):
        base_dir = '/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_accum'
        model_path = f"{base_dir}/{ds_dir}"
        json_file = os.path.join(model_path, 'final_training_summary.json')
        if not os.path.isfile(json_file):
            json_path.append(None)
            continue
    json_path.append(json_file)
print(f"loaded {len(json_path)} json paths")

if 'best accum' not in df.columns:
    df['best accum'] = None

updated = 0
missing_key_count = 0
no_match_count = 0
for i, path in enumerate(json_path):
    if path is None:
        continue
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Try several possible keys for balanced accuracy
        avg_block = data.get('averages', {}) or {}
        ba = (
            avg_block.get('balanced_accuracy')
            or avg_block.get('balanced_accuracy_common')
            or avg_block.get('balanced_accuracy_macro')
            or data.get('balanced_accuracy')  # fallback if stored top-level
        )
        if ba is not None:
            # Match on normalized dataset name against df['dataset']
            ds_norm = datasets_norm[i]
            # Normalize df dataset column once (strip + replace potential stray slashes)
            if '_df_dataset_norm' not in locals():
                df['_df_dataset_norm'] = df['dataset'].astype(str).str.strip()
            mask = df['_df_dataset_norm'] == ds_norm
            if mask.any():
                df.loc[mask, 'best accum'] = ba
                updated += int(mask.sum())
            else:
                no_match_count += 1
                print(f"[WARN] No DataFrame match for dataset '{ds_norm}' (from path {path})")
        else:
            missing_key_count += 1
            print(f"[WARN] Balanced accuracy keys missing in summary for '{datasets_norm[i]}' -> {path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

print(f"Updated 'best accum' for {updated} rows out of {len(datasets_norm)} datasets")
print(f"Datasets with missing BA key: {missing_key_count}")
print(f"Datasets with no DataFrame match: {no_match_count}")

df.drop(columns=['_df_dataset_norm'], errors='ignore', inplace=True)
df.to_csv(CSV_PATH, index=False)