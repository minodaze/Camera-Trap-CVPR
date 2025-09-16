import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

with open('ad_accu.list', 'r') as f:
    data_set = [line.strip() for line in f.readlines()]

print(data_set)

for ds in data_set:
    # /fs/scratch/PAS2099/camera-trap-final/eval_logs/{ds}/eval_full_accu/bioclip2/full_text_head_merge_factor_1/log/eval_only_summary.json
    ds = ds.replace('/', '_', 1)
    json_path = f'/fs/scratch/PAS2099/camera-trap-final/eval_logs/{ds}/eval_full_accu/bioclip2/full_text_head_merge_factor_1/log/eval_only_summary.json'
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        continue
    n_ckp = len(json_data)
    print(f"Number of checkpoints for {ds}: {n_ckp}")
    
    # Calculate which checkpoints to use (1/3, 2/3, 3/3 of total)
    ckp_list = [n*n_ckp // 3 for n in [1, 2, 3]]
    
    plt.figure(figsize=(10, 6))
    
    # Plot 3 curves, one for each checkpoint in ckp_list
    for i, ckp in enumerate(ckp_list):
        ckp_key = f'ckp_{ckp}'
        
        # Get all checkpoint numbers for x-axis
        all_ckps = list(range(1, n_ckp + 1))
        balanced_accuracies = []
        
        # Extract balanced accuracy for this specific checkpoint across all evaluations
        for eval_ckp in all_ckps:
            eval_key = f'ckp_{eval_ckp}'
            if ckp_key in json_data and eval_key in json_data[ckp_key]:
                balanced_acc = json_data[ckp_key][eval_key]['balanced_accuracy']
                balanced_accuracies.append(balanced_acc)
            else:
                balanced_accuracies.append(None)  # Handle missing data
        
        # Plot the curve
        if i != 2:
            label = f'{i+1}/3 Checkpoint'
        else:
            label = f'All Checkpoint'
        plt.plot(all_ckps, balanced_accuracies, marker='o', label=label, linewidth=2)


    plt.xlabel('Evaluation Checkpoint')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Balanced Accuracy Curves for {ds.replace("_", " ").title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'plots/{ds}_balanced_accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

