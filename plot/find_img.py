import pandas as pd
import json
import matplotlib.pyplot as plt

with open('plot/plot.txt', 'r') as file:
    dataset = file.read().splitlines()

for ds in dataset:
    ds = ds.replace("/", "_", 1)
    zs_path = f'/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/{ds}/zs/bioclip2/full_text_head/log/final_image_level_predictions.json'
    ub_ce_path = f'/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/{ds}/upper_bound_ce_loss/bioclip2/full_text_head/log/final_image_level_predictions.json'
    ub_bsm_path = f'/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/{ds}/upper_bound_bsm_loss/bioclip2/full_text_head/log/final_image_level_predictions.json'
    ub_lora_path = f'/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/{ds}/upper_bound_bsm_loss/bioclip2/lora_8_text_head/log/final_image_level_predictions.json'
    with open(zs_path, 'r') as f:
        zs_data = json.load(f)
    with open(ub_ce_path, 'r') as f:
        ub_ce_data = json.load(f)
    with open(ub_bsm_path, 'r') as f:
        ub_bsm_data = json.load(f)
    with open(ub_lora_path, 'r') as f:
        ub_lora_data = json.load(f)
    ckp_num = len(zs_data) - 1

    common_false_pred = {}
    n_img = {}
    for ckp, v in ub_ce_data.items():
        if ckp == 'stats':
            continue
        incorrect_preds = v['incorrect']
        num_incorrect = sum(len(items) for items in incorrect_preds.values())
        current_set = set()
        n = 0
        for label, items in incorrect_preds.items():
            for item in items:
                current_set.add(item['file_path'])
        for label, items in v['correct'].items():
            for item in items:
                n += 1
        common_false_pred[ckp] = current_set
        n_img[ckp] = n + num_incorrect

    for ckp, v in ub_bsm_data.items():
        if ckp == 'stats':
            continue
        incorrect_preds = v['incorrect']
        num_incorrect = sum(len(items) for items in incorrect_preds.values())
        current_set = set()
        for label, items in incorrect_preds.items():
            for item in items:
                current_set.add(item['file_path'])
        common_false_pred[ckp] = common_false_pred[ckp].intersection(current_set)
    for ckp, v in ub_lora_data.items():
        if ckp == 'stats':
            continue
        incorrect_preds = v['incorrect']
        num_incorrect = sum(len(items) for items in incorrect_preds.values())
        current_set = set()
        for label, items in incorrect_preds.items():
            for item in items:
                current_set.add(item['file_path'])
        common_false_pred[ckp] = common_false_pred[ckp].intersection(current_set)
    for ckp, v in zs_data.items():
        if ckp == 'stats':
            continue
        incorrect_preds = v['incorrect']
        num_incorrect = sum(len(items) for items in incorrect_preds.values())
        current_set = set()
        for label, items in incorrect_preds.items():
            for item in items:
                current_set.add(item['file_path'])
        common_false_pred[ckp] = common_false_pred[ckp].intersection(current_set)
    
    for ckp in common_false_pred:
        common_false_pred[ckp] = list(common_false_pred[ckp])
        common_false_pred[ckp].append(f"Total Evaluated images: {n_img[ckp]}")
        common_false_pred[ckp].append(f"Total Common False Predictions: {len(common_false_pred[ckp])-1}")

    # import pdb; pdb.set_trace()
    output_dir = f'/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/{ds}'
    with open(f'{output_dir}/common_false_predictions.json', 'w') as f:
        json.dump(common_false_pred, f, indent=2)
