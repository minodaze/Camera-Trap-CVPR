import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

data_dirs = [
    "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/idaho_idaho_76/eval_interpolation/lr_0.000025/2025-07-16-23-60-06/bioclip2",
    "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/idaho_idaho_85/eval_interpolation/lr_0.000025/2025-07-16-23-5f-1f/bioclip2",
    "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/idaho_idaho_103/eval_interpolation/lr_0.000025/2025-07-16-23-7f-b3/bioclip2",
    "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/na_na_archbold_FL-23/eval_interpolation/lr_0.000025/2025-07-16-23-52-4b/bioclip2",
    "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/nz_nz_EFH_HCAMD07/eval_interpolation/lr_0.000025/2025-07-16-23-4d-f5/bioclip2",
]

data_json = {
    "idaho_idaho_76": {
        "0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0
    }, 
    "idaho_idaho_85": {
        "0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0
    }, 
    "idaho_idaho_103": {
        "0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0
    }, 
    "na_na_archbold_FL-23": {
        "0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0
    }, 
    "nz_nz_EFH_HCAMD07": {
        "0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0
    }
}
for dir in data_dirs:
    assert os.path.exists(dir), f"Directory {dir} does not exist."
    for i in range (1, 10):
        json_file = os.path.join(dir, f"full_text_head_interpolation_model_0.{i}/log/eval_only_summary.json")
        assert os.path.exists(json_file), f"JSON file {json_file} does not exist."
        with open(json_file, "r") as f:
            json_data = json.load(f)
        balanced_accs = []
        for ckp, v in json_data.items():
            balanced_accs.append(v["balanced_accuracy"])
        balanced_acc = sum(balanced_accs) / len(balanced_accs)
        dataset_name = dir.split("/")[-5]  # Extract dataset name from path
        data_json[dataset_name][f"0.{i}"] = balanced_acc

for k, v in data_json.items():
    print(f"{k}: {v}")

csv_path = "/users/PAS2099/mino/ICICLE/CL + Animal Trap - ML Studies Dataset.csv"
df = pd.read_csv(csv_path)

valid_datasets = df['dataset'].apply(lambda x: x.replace("/", "_", 1) in data_json)
df = df[valid_datasets]

df['coefficient-0.1'] = 0.0
df['coefficient-0.2'] = 0.0
df['coefficient-0.3'] = 0.0
df['coefficient-0.4'] = 0.0
df['coefficient-0.5'] = 0.0
df['coefficient-0.6'] = 0.0
df['coefficient-0.7'] = 0.0
df['coefficient-0.8'] = 0.0
df['coefficient-0.9'] = 0.0
for idx, row in df.iterrows():
    x = row['dataset']
    x = x.replace("/", "_", 1)  # Convert only the first "/" to "_"
    if x in data_json:
        df.at[idx, 'coefficient-0.1'] = data_json[x]['0.1'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.2'] = data_json[x]['0.2'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.3'] = data_json[x]['0.3'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.4'] = data_json[x]['0.4'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.5'] = data_json[x]['0.5'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.6'] = data_json[x]['0.6'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.7'] = data_json[x]['0.7'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.8'] = data_json[x]['0.8'] if x in data_json else np.nan
        df.at[idx, 'coefficient-0.9'] = data_json[x]['0.9'] if x in data_json else np.nan
    else:
        print(f"Warning: Dataset {x} not found in interpolation results.")



import matplotlib.pyplot as plt

# Grab every interpolation-coefficient column and sort them numerically
coeff_cols = sorted(
    [c for c in df.columns if c.startswith("coefficient-")],
    key=lambda s: float(s.split("-")[1])
)
coeff_values = [float(c.split("-")[1]) for c in coeff_cols]   # e.g. [0.1, 0.2, …]

plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    dataset_name = row["dataset"]
    x = [0] + coeff_values + [1]                     # α: 0 (zero-shot) → … → 1 (upper-bound)
    y = [row["ub"]] + [row[c] for c in coeff_cols] + [row["zs"]]
    plt.plot(x, y, marker="o", label=dataset_name)

plt.xlabel("Interpolation coefficient (α)")
plt.ylabel("Balanced Accuracy")
plt.title("Upper-bound → Interpolated → Zero-shot balanced accuracy per dataset")
plt.ylim(0, 1)
plt.grid(True)
plt.legend(fontsize="small", ncol=2)                 # tweak ncol as you like
plt.tight_layout()
plt.savefig("CL + Animal Trap - ML Studies Dataset with interpolation.png")
