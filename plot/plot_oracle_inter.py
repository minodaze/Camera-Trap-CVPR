import pandas as pd
import matplotlib.pyplot as plt
import os
import json

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - (cvpr) Oracle BSM _ LoRA Emperical (2).csv')
with open('/users/PAS2099/mino/ICICLE/plot/oracle.txt', 'r') as file:
    datasets = [l.strip() for l in file.read().splitlines() if l.strip() and not l.strip().startswith('#')]
out_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - (cvpr) Oracle BSM _ LoRA Emperical with oracles interpolation.csv'
base_dir = '/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_inter/'

def load_oracle_ba_for(ds: str, ratio: float):
    """Return balanced_accuracy for a dataset at a given ratio, trying both raw and '_' normalized dataset folder names."""
    # Primary: use ds as-is
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_inter/oracle_inter/na_na_archbold_FL-09/bioclip2/full_text_head_interpolation_model_0.1/log
    ds = ds.replace("/", "_")
    model_path = f"{base_dir}/oracle_inter/{ds}/bioclip2/full_text_head_interpolation_model_{ratio}/log"
    json_path = os.path.join(model_path, 'eval_only_summary.json')
    if not os.path.isfile(json_path):
        print(f"[WARN] Oracle JSON not found for dataset {ds} at ratio {ratio}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['average']['balanced_accuracy']
    except Exception:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data['averages']['balanced_accuracy']
        except Exception:
            return None

def load_bsm_oracle_ba_for(ds: str, ratio: float):
    """Return balanced_accuracy for a dataset at a given ratio, trying both raw and '_' normalized dataset folder names."""
    # Primary: use ds as-is
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_inter/oracle_inter/na_na_archbold_FL-09/bioclip2/full_text_head_interpolation_model_0.1/log
    ds = ds.replace("/", "_")
    model_path = f"{base_dir}/bsm_oracle_inter/{ds}/bioclip2/full_text_head_interpolation_model_{ratio}/log"
    json_path = os.path.join(model_path, 'eval_only_summary.json')
    if not os.path.isfile(json_path):
        print(f"[WARN] BSM Oracle JSON not found for dataset {ds} at ratio {ratio}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['average']['balanced_accuracy']
    except Exception:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data['averages']['balanced_accuracy']
        except Exception:
            return None

def load_lora_oracle_ba_for(ds: str, ratio: float):
    """Return balanced_accuracy for a dataset for LoRA Oracle."""
    ds = ds.replace("/", "_")
    # camera-trap-CVPR-logs/oracle_inter/lora_oracle_inter/APN_APN_U64C/bioclip2/lora_8_text_head_lora_interpolate_0.1/log
    model_path = f"{base_dir}/lora_oracle_inter/{ds}/bioclip2/lora_8_text_head_lora_interpolate_{ratio}/log"
    json_path = os.path.join(model_path, 'eval_only_summary.json')
    if not os.path.isfile(json_path):
        print(f"[WARN] LoRA Oracle JSON not found for dataset {ds} at ratio {ratio}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['average']['balanced_accuracy']
    except Exception:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data['averages']['balanced_accuracy']
        except Exception:
            return None

def load_best_oracle_ba_for(ds: str, ratio: float):
    """Return balanced_accuracy for a dataset for Best Oracle."""
    ds = ds.replace("/", "_")
    # camera-trap-CVPR-logs/oracle_inter/best_oracle_inter/APN_APN_U64C/bioclip2/lora_8_text_head_lora_interpolate_0.1/log
    model_path = f"{base_dir}/best_oracle_inter/{ds}/bioclip2/lora_8_text_head_lora_interpolate_{ratio}/log"
    json_path = os.path.join(model_path, 'eval_only_summary.json')
    if not os.path.isfile(json_path):
        print(f"[WARN] Best Oracle JSON not found for dataset {ds} at ratio {ratio}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['average']['balanced_accuracy']
    except Exception:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data['averages']['balanced_accuracy']
        except Exception:
            return None

ratios = [round(0.1 * i, 2) for i in range(1, 10)]
oracle_data_per_dataset = {}
lora_oracle_data_per_dataset = {}
bsm_oracle_data_per_dataset = {}
best_oracle_data_per_dataset = {}
for ds in datasets:
    oracle_values = []
    lora_oracle_values = []
    bsm_oracle_values = []
    best_oracle_values = []
    for r in ratios:
        oracle_values.append(load_oracle_ba_for(ds, r))
        lora_oracle_values.append(load_lora_oracle_ba_for(ds, r))
        bsm_oracle_values.append(load_bsm_oracle_ba_for(ds, r))
        best_oracle_values.append(load_best_oracle_ba_for(ds, r))
    oracle_data_per_dataset[ds] = oracle_values
    lora_oracle_data_per_dataset[ds] = lora_oracle_values
    bsm_oracle_data_per_dataset[ds] = bsm_oracle_values
    best_oracle_data_per_dataset[ds] = best_oracle_values

df['Oracle Interpolation 0.8'] = 0.0
df['LoRA Oracle Interpolation 0.8'] = 0.0
df['BSM Oracle Interpolation 0.8'] = 0.0
df['Best Oracle Interpolation 0.8'] = 0.0
for idx, row in df.iterrows():
    dataset_name = row['dataset']
    if dataset_name not in oracle_data_per_dataset:
        continue
    max_oracle_ba = 0.0
    max_lora_oracle_ba = 0.0
    max_bsm_oracle_ba = 0.0
    max_best_oracle_ba = 0.0
    for i, ratio in enumerate(ratios):
        max_oracle_ba = max(oracle_data_per_dataset[dataset_name][i], max_oracle_ba)
        max_lora_oracle_ba = max(lora_oracle_data_per_dataset[dataset_name][i], max_lora_oracle_ba)
        max_bsm_oracle_ba = max(bsm_oracle_data_per_dataset[dataset_name][i], max_bsm_oracle_ba)
        max_best_oracle_ba = max(best_oracle_data_per_dataset[dataset_name][i], max_best_oracle_ba)
    df.at[idx, 'Oracle Interpolation'] = max_oracle_ba
    df.at[idx, 'LoRA Oracle Interpolation'] = max_lora_oracle_ba
    df.at[idx, 'BSM Oracle Interpolation'] = max_bsm_oracle_ba
    df.at[idx, 'Best Oracle Interpolation'] = max_best_oracle_ba
    df.at[idx, 'Oracle Interpolation 0.8'] = oracle_data_per_dataset[dataset_name][7]
    df.at[idx, 'LoRA Oracle Interpolation 0.8'] = lora_oracle_data_per_dataset[dataset_name][7]
    df.at[idx, 'BSM Oracle Interpolation 0.8'] = bsm_oracle_data_per_dataset[dataset_name][7]
    df.at[idx, 'Best Oracle Interpolation 0.8'] = best_oracle_data_per_dataset[dataset_name][7]

df.to_csv(out_path, index=False)