import pandas as pd
import matplotlib.pyplot as plt
import os
import json

base_dir = '/fs/ess/PAS2099/camera-trap-CVPR-logs'

dataset_path = '/users/PAS2099/mino/ICICLE/uselist/rare.txt'

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Rare.csv')

out_csv = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Filled Rare.csv'

def load_rare_zs(ds: str):
    """Return balanced_accuracy for a dataset at a given ratio for zero-shot model."""
    ds = ds.replace("/", "_")
    model_path = f"{base_dir}/rare_zs/{ds}/bioclip2/full_text_head/log"
    json_path = os.path.join(model_path, 'final_training_summary.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['averages']['balanced_accuracy']

def load_rare_best_oracle(ds: str):
    """Return balanced_accuracy for a dataset at a given ratio for best oracle model."""
    ds = ds.replace("/", "_")
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/rare_best_oracle/ENO_ENO_C04/bioclip2/lora_8_text_head/log
    model_path = f"{base_dir}/rare_best_oracle/{ds}/bioclip2/lora_8_text_head/log"
    json_path = os.path.join(model_path, 'final_training_summary.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['averages']['balanced_accuracy']

def load_rare_best_accum(ds: str):
    """Return balanced_accuracy for a dataset at a given ratio for best accum model."""
    ds = ds.replace("/", "_")
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/rare_best_accum/serengeti_serengeti_D02/bioclip2/lora_8_text_head/all/log
    model_path = f"{base_dir}/rare_best_accum/{ds}/bioclip2/lora_8_text_head/all/log"
    try:
        json_path = os.path.join(model_path, 'final_training_summary.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['averages']['balanced_accuracy']
    except Exception:
        return 0.0

def load_best_accum(ds: str):
    """Return balanced_accuracy for a dataset at a given ratio for best accum model."""
    ds = ds.replace("/", "_")
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/best_accum/serengeti_serengeti_D02/bioclip2/lora_8_text_head/all/log
    json_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/{ds}/bioclip2/lora_8_text_head/all/log/final_training_summary.json'
    if not os.path.exists(json_path):
        json_path = f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_accum/{ds}/final_training_summary.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    # import pdb; pdb.set_trace()
    return data['averages']['balanced_accuracy']

def load_best_oracle(ds: str):
    """Return balanced_accuracy for a dataset at a given ratio for best oracle model."""
    ds = ds.replace("/", "_")
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/best_oracle/ENO_ENO_C04/bioclip2/lora_8_text_head/log
    json_path = f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle/{ds}/final_training_summary.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['averages']['balanced_accuracy']

with open(dataset_path, 'r') as file:
    datasets = [l.strip() for l in file.read().splitlines()]

for ds in datasets:
    zs_ba = load_rare_zs(ds)
    best_oracle_ba = load_rare_best_oracle(ds)
    best_accum_ba = load_rare_best_accum(ds)
    df.loc[df['dataset'] == ds, 'rare zs'] = zs_ba
    df.loc[df['dataset'] == ds, 'rare best oracle'] = best_oracle_ba
    df.loc[df['dataset'] == ds, 'rare best accum'] = best_accum_ba
    df.loc[df['dataset'] == ds, 'best oracle'] = load_best_oracle(ds)
    df.loc[df['dataset'] == ds, 'best accum'] = load_best_accum(ds)
    
df.to_csv(out_csv, index=False)