import pandas as pd
import os
import json

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - CLIP _ Siglip2 ZS.csv')
out_csv_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - CLIP _ Siglip2 ZS_filled.csv'

with open('/users/PAS2099/mino/ICICLE/plot/zs_list.txt', 'r') as file:
    datasets = [line.strip() for line in file.readlines()]

for ds in datasets:
    row = df.loc[df['dataset'] == ds]
    ds_formatted = ds.replace("/", "_")
    json_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/zs/{ds_formatted}/openai-ViT-L-14/full_text_head/log/final_training_summary.json'
    # Skip if file is missing or empty
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        continue
    if os.path.getsize(json_path) == 0:
        print(f"File is empty, skipping: {json_path}")
        continue
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON ({e}) in: {json_path}")
        continue
    balanced_acc = data['averages']['balanced_accuracy']
    # Write back into the original DataFrame (not a copy)
    df.loc[df['dataset'] == ds, 'CLIP'] = balanced_acc

df.to_csv(out_csv_path, index=False)