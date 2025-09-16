import pandas as pd
import json
import matplotlib.pyplot as plt

with open('plot/plot.txt', 'r') as file:
    dataset = file.read().splitlines()

plt.figure(figsize=(12, 8))

for ds in dataset:
    ds_name = ds.replace("/", "_", 1)
    zs_path = f'/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/{ds_name}/zs/bioclip2/full_text_head/log/final_image_level_predictions.json'
    
    try:
        with open(zs_path, 'r') as f:
            zs_data = json.load(f)
        
        ckp_num = len(zs_data) - 1
        zs_ckp_conf = []
        
        for ckp, v in zs_data.items():
            if ckp == 'stats':
                uniform_dist = 1 / v['num_cls']
            else:
                zs_ckp_conf.append(v['avg_confidence'])

        # Plot the confidence curve for this dataset
        
        plt.plot(range(1, ckp_num + 1), zs_ckp_conf, marker="o", linestyle="-", label=f"{ds}")
        plt.xlabel("Checkpoint")
        plt.ylabel("Zero-Shot Average Confidence per checkpoint")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plots/zs_confidence_{ds}.png', bbox_inches='tight', dpi=300)
        plt.show()
        print(f"Plot saved as 'plots/zs_confidence_{ds}.png'")

    except FileNotFoundError:
        print(f"Warning: File not found for dataset {ds}")
        continue
    except Exception as e:
        print(f"Error processing dataset {ds}: {e}")
        continue
