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
    zs_balanced_acc = []
    zs_ckp_conf = []
    ub_ce_balanced_acc = []
    ub_ce_ckp_conf = []
    ub_bsm_balanced_acc = []
    ub_bsm_ckp_conf = []
    ub_lora_bsm_balanced_acc = []
    ub_lora_bsm_ckp_conf = []
    for ckp, v in ub_ce_data.items():
        if ckp == 'stats':
            continue
        correct_preds = v['correct']
        incorrect_preds = v['incorrect']
        num_correct = sum(len(items) for items in correct_preds.values())
        num_incorrect = sum(len(items) for items in incorrect_preds.values())
        ckp_conf = 0
        for label, items in correct_preds.items():
            for item in items:
                confidence = item['confidence']
                ckp_conf += confidence
        for label, items in incorrect_preds.items():
            for item in items:
                confidence = item['confidence']
                ckp_conf += confidence
        ckp_conf /= max(1, num_correct + num_incorrect)
        # Process correct predictions if needed
        ub_ce_balanced_acc.append(v['balanced_accuracy'])
        ub_ce_ckp_conf.append(ckp_conf)
    for ckp, v in ub_bsm_data.items():
        if ckp == 'stats':
            continue
        correct_preds = v['correct']
        incorrect_preds = v['incorrect']
        num_correct = sum(len(items) for items in correct_preds.values())
        num_incorrect = sum(len(items) for items in incorrect_preds.values())
        ckp_conf = 0
        for label, items in correct_preds.items():
            for item in items:
                confidence = item['confidence']
                ckp_conf += confidence
        for label, items in incorrect_preds.items():
            for item in items:
                confidence = item['confidence']
                ckp_conf += confidence
        ckp_conf /= max(1, num_correct + num_incorrect)
        # Process correct predictions if needed
        ub_bsm_balanced_acc.append(v['balanced_accuracy'])
        ub_bsm_ckp_conf.append(ckp_conf)
    for ckp, v in ub_lora_data.items():
        if ckp == 'stats':
            continue
        correct_preds = v['correct']
        incorrect_preds = v['incorrect']
        num_correct = sum(len(items) for items in correct_preds.values())
        num_incorrect = sum(len(items) for items in incorrect_preds.values())
        ckp_conf = 0
        for label, items in correct_preds.items():
            for item in items:
                confidence = item['confidence']
                ckp_conf += confidence
        for label, items in incorrect_preds.items():
            for item in items:
                confidence = item['confidence']
                ckp_conf += confidence
        ckp_conf /= max(1, num_correct + num_incorrect)
        # Process correct predictions if needed
        ub_lora_bsm_balanced_acc.append(v['balanced_accuracy'])
        ub_lora_bsm_ckp_conf.append(ckp_conf)
    for ckp, v in zs_data.items():
        if ckp == 'stats':
            continue
        zs_balanced_acc.append(v['balanced_accuracy'])
        zs_ckp_conf.append(v['avg_confidence'])
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, ckp_num + 1), zs_balanced_acc, marker="o", linestyle="-", label="ZS Balanced Accuracy")
    plt.plot(range(1, ckp_num + 1), zs_ckp_conf, marker="s", linestyle="--", label="ZS Average Confidence")
    plt.plot(range(1, ckp_num + 1), ub_ce_balanced_acc, marker="o", linestyle="-", label="Oracle-CE Balanced Accuracy")
    plt.plot(range(1, ckp_num + 1), ub_ce_ckp_conf, marker="s", linestyle="--", label="Oracle-CE Average Confidence")
    plt.plot(range(1, ckp_num + 1), ub_lora_bsm_balanced_acc, marker="o", linestyle="-", label="Oracle-LoRA-BSM Balanced Accuracy")
    plt.plot(range(1, ckp_num + 1), ub_lora_bsm_ckp_conf, marker="s", linestyle="--", label="Oracle-LoRA-BSM Average Confidence")
    plt.plot(range(1, ckp_num + 1), ub_bsm_balanced_acc, marker="o", linestyle="-", label="Oracle-BSM Balanced Accuracy")
    plt.plot(range(1, ckp_num + 1), ub_bsm_ckp_conf, marker="s", linestyle="--", label="Oracle-BSM Average Confidence")

    plt.xlabel("Checkpoint")
    plt.ylabel("Value")
    plt.title(f"{ds} Balanced Accuracy and Confidence per Checkpoint")
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(f'plots/plot_{ds}_balanced_acc_conf.png', bbox_inches='tight')

