import pandas as pd
import os
import json

csv_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Final Imbalance loss _ PEFT comparison (3).csv'
output_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR reload.csv'
dataset_path = '/users/PAS2099/mino/ICICLE/plot/oracle.txt'

df = pd.read_csv(csv_path)

with open(dataset_path, 'r') as f:
    datasets = f.read().splitlines()

for dataset in datasets:
    row = df.loc[df['dataset'].eq(dataset)]
    best_oracle_path = f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle//{dataset.replace("/", "_")}/final_training_summary.json'
    try:
        with open(best_oracle_path, 'r') as f:
            best_oracle_data = json.load(f)
            best_oracle_acc = best_oracle_data['averages']['balanced_accuracy']
            df.loc[df['dataset'].eq(dataset), 'Best Oracle'] = best_oracle_acc
    except FileNotFoundError:
        print(f"File not found for dataset: {dataset}")

df.to_csv(output_path, index=False)