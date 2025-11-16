import pandas as pd
import os
import json

csv_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - (cvpr) Oracle BSM _ LoRA Emperical.csv'
output_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR loaded - (cvpr) Oracle BSM _ LoRA Emperical.csv'
dataset_path = '/users/PAS2099/mino/ICICLE/plot/oracle.txt'

df = pd.read_csv(csv_path)

with open(dataset_path, 'r') as f:
    datasets = f.read().splitlines()

for dataset in datasets:
    row = df.loc[df['dataset'].eq(dataset)]
    oracle_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/rere_oracle_20/oracle/{dataset}/bioclip2/full_text_head/all/log/final_training_summary.json'
    bsm_oracle_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/rere_oracle_20/bsm_oracle/{dataset}/bioclip2/full_text_head/all/log/final_training_summary.json'
    lora_oracle_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/rere_oracle_20/lora_oracle/{dataset}/bioclip2/lora_8_text_head/all/log/final_training_summary.json'
    try:
        with open(oracle_path, 'r') as f:
            oracle_data = json.load(f)
        with open(bsm_oracle_path, 'r') as f:
            bsm_oracle_data = json.load(f)
        with open(lora_oracle_path, 'r') as f:
            lora_oracle_data = json.load(f)
        oracle_ba = oracle_data['averages']['balanced_accuracy']
        bsm_oracle_ba = bsm_oracle_data['averages']['balanced_accuracy']
        lora_oracle_ba = lora_oracle_data['averages']['balanced_accuracy']
        df.loc[row.index, 'Oracle'] = oracle_ba
        df.loc[row.index, 'Oracle + BSM'] = bsm_oracle_ba
        df.loc[row.index, 'Oracle + LoRA'] = lora_oracle_ba
    except FileNotFoundError:
        print(f"File not found for dataset: {dataset}")

df.to_csv(output_path, index=False)