
import matplotlib.pyplot as plt
import json
import os

# Read dataset identifiers (one per line). Lines can be like "serengeti/serengeti_S11" or already underscored.
with open('/users/PAS2099/mino/ICICLE/eval_best_accum_lora_dataset.txt', 'r') as file:
    datasets = [l.strip() for l in file.read().splitlines() if l.strip() and not l.strip().startswith('#')]

# Ratios from 0.1 to 0.9 (inclusive)
ratios = [round(0.1 * i, 2) for i in range(1, 10)]

base_dir = '/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_inter'

def load_ba_for(ds: str, ratio: float):
    """Return balanced_accuracy for a dataset at a given ratio, trying both raw and '_' normalized dataset folder names."""
    # Primary: use ds as-is
    ds = ds.replace('/', '_')
    model_path = f"{base_dir}/{ds}/bioclip2/lora_8_text_head_lora_interpolate_{ratio}/log"
    json_path = os.path.join(model_path, 'eval_only_summary.json')
    if not os.path.isfile(json_path):
        print(f"[WARN] JSON not found for dataset {ds} at ratio {ratio}")
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

# Collect data per dataset
data_per_dataset = {}
for ds in datasets:
    values = []
    for r in ratios:
        values.append(load_ba_for(ds, r))
    data_per_dataset[ds] = values

# Plot all datasets on one figure
plt.figure(figsize=(10, 6))
for ds, vals in data_per_dataset.items():
    xs = [r for r, v in zip(ratios, vals) if v is not None]
    ys = [v for v in vals if v is not None]
    if not xs:
        continue
    plt.plot(xs, ys, marker='o', linewidth=2, label=ds)

plt.title('LoRA interpolation: Balanced Accuracy vs Merge Ratio')
plt.xlabel('LoRA Merge Ratio')
plt.ylabel('Balanced Accuracy')
plt.xticks(ratios)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(
    title='Dataset',
    fontsize=6,
    title_fontsize=7,
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    ncol=2,
    markerscale=0.8,
    handlelength=1.2,
    columnspacing=0.8,
    borderaxespad=0.5,
    frameon=False
)
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots2/lora_interpolation_all_datasets.png', dpi=200)
plt.close()

import pandas as pd
df = pd.read_csv('other/camera-trap-CVPR - (Correct) Final Best Accum List.csv')
# import pdb; pdb.set_trace()
for idx, row in df.iterrows():
    dataset_name = row['dataset']
    if dataset_name not in data_per_dataset:
        continue
    value = 0.0
    max_ratio = 0.0
    for i, ratio in enumerate(ratios):
        try:
            value = max(data_per_dataset[dataset_name][i], value)
            if data_per_dataset[dataset_name][i] == value:
                max_ratio = ratio
        except Exception:
            import pdb; pdb.set_trace()
            raise ValueError(f"Error accessing data for dataset {dataset_name} at ratio index {i}")
    df.loc[idx, 'best accum + best inter ratio'] = value
    df.loc[idx, 'best accum interpolate - 0.8'] = data_per_dataset[dataset_name][7]  # ratio 0.8 is index 7

df.to_csv('csv/camera-trap-CVPR - (Correct) lora-inter best ratio Final Best Accum List.csv', index=False)
print(f"Updated CSV with interpolation results to other/camera-trap-CVPR - (Correct) lora-inter best ratio Final Best Accum List.csv.")
