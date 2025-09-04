import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

data_names = [
    "orinoquia/orinoquia_N25",
    "na/na_archbold_FL-32",
    "serengeti/serengeti_F05",
    "serengeti/serengeti_E08",
    "nz/nz_EFH_HCAMD07",
    "nz/nz_EFD_DCAMF09",
    "serengeti/serengeti_D12",
    "MAD/MAD_A05",
    "caltech/caltech_88",
    "serengeti/serengeti_L09",
    "serengeti/serengeti_E03",
    "KGA/KGA_KHOGB07",
    "nz/nz_EFH_HCAMC01",
    "KGA/KGA_KHOLA08",
    "orinoquia/orinoquia_N29",
    "nz/nz_EFH_HCAMD03",
    "nz/nz_EFH_HCAMB10",
    "KGA/KGA_KHOLA03",
    "nz/nz_PS1_CAM6213",
    "nz/nz_EFD_DCAMB02"
]

def dataset_to_evalpath(dataset):
    """
    Convert dataset name to evaluation path following the pattern:
    /fs/scratch/PAS2099/camera-trap-final/logs/{dataset_underscore}/upper_bound_bsm_loss/lr_0.000025/bioclip2/full_text_head/log/eval_only_summary.json
    
    Args:
        dataset (str): Dataset name like "orinoquia/orinoquia_N25"
    
    Returns:
        str: Full evaluation path
    """
    # /fs/scratch/PAS2099/camera-trap-final/eval_logs/nz_nz_EFH_HCAMB10/eval_lora_bsm_ub_interpolation/bioclip2/lora_8_text_head_merge_factor_0.5
    # Convert slashes to underscores for the path
    dataset_underscore = dataset.replace("/", "_", 1)
    eval_paths = {}
    for setting in ['full_ce', 'full_bsm', 'lora_bsm', 'lora']:
        if 'lora' in setting:
            eval_path = f"/fs/scratch/PAS2099/camera-trap-final/eval_logs/{dataset_underscore}/eval_{setting}_ub_interpolation/bioclip2/lora_8_text_head_merge_factor_0.5/log/eval_only_summary.json"
        else:
            eval_path = f"/fs/scratch/PAS2099/camera-trap-final/eval_logs/{dataset_underscore}/eval_{setting}_ub_interpolation/bioclip2/full_text_head_interpolation_model_0.5_merge_factor_1/log/eval_only_summary.json"
        eval_paths[setting] = eval_path
    return eval_paths

import pandas as pd
df = pd.read_csv("CL + Animal Trap - Final ML Study Dataset.csv")

import json
for dataset in data_names:
    eval_paths = dataset_to_evalpath(dataset)
    for setting, eval_path in eval_paths.items():
        try:
            with open(eval_path, 'r') as f:
                data_json = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {eval_path}")
            continue
        balanced_accuracys = []
        for k, v in data_json.items():
            balanced_accuracys.append(v['balanced_accuracy'])
        avg_ba = sum(balanced_accuracys) / len(balanced_accuracys)
        if setting == 'full_ce':
            df.loc[df['dataset'] == dataset, 'UB-FT-CE-Interpolation-0.5'] = avg_ba
        elif setting == 'full_bsm':
            df.loc[df['dataset'] == dataset, 'UB-FT-bsm-Interpolation-0.5'] = avg_ba
        elif setting == 'lora':
            df.loc[df['dataset'] == dataset, 'UB-LoRA-CE-interpolation-0.5'] = avg_ba
        elif setting == 'lora_bsm':
            df.loc[df['dataset'] == dataset, 'UB-LoRA-bsm-interpolation-0.5'] = avg_ba

df.to_csv("CL + Animal Trap - Final ML Study Dataset with UB Interpolation.csv", index=False)

# for k, v in data_json.items():
#     print(f"{k}: {v}")

# csv_path = "/users/PAS2099/mino/ICICLE/CL + Animal Trap - ML Studies Dataset.csv"
# df = pd.read_csv(csv_path)

# valid_datasets = df['dataset'].apply(lambda x: x.replace("/", "_", 1) in data_json)
# df = df[valid_datasets]

# df['coefficient-0.1'] = 0.0
# df['coefficient-0.2'] = 0.0
# df['coefficient-0.3'] = 0.0
# df['coefficient-0.4'] = 0.0
# df['coefficient-0.5'] = 0.0
# df['coefficient-0.6'] = 0.0
# df['coefficient-0.7'] = 0.0
# df['coefficient-0.8'] = 0.0
# df['coefficient-0.9'] = 0.0
# for idx, row in df.iterrows():
#     x = row['dataset']
#     x = x.replace("/", "_", 1)  # Convert only the first "/" to "_"
#     if x in data_json:
#         df.at[idx, 'coefficient-0.1'] = data_json[x]['0.1'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.2'] = data_json[x]['0.2'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.3'] = data_json[x]['0.3'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.4'] = data_json[x]['0.4'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.5'] = data_json[x]['0.5'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.6'] = data_json[x]['0.6'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.7'] = data_json[x]['0.7'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.8'] = data_json[x]['0.8'] if x in data_json else np.nan
#         df.at[idx, 'coefficient-0.9'] = data_json[x]['0.9'] if x in data_json else np.nan
#     else:
#         print(f"Warning: Dataset {x} not found in interpolation results.")



# import matplotlib.pyplot as plt

# # Grab every interpolation-coefficient column and sort them numerically
# coeff_cols = sorted(
#     [c for c in df.columns if c.startswith("coefficient-")],
#     key=lambda s: float(s.split("-")[1])
# )
# coeff_values = [float(c.split("-")[1]) for c in coeff_cols]   # e.g. [0.1, 0.2, …]

# plt.figure(figsize=(10, 6))

# for _, row in df.iterrows():
#     dataset_name = row["dataset"]
#     x = [0] + coeff_values + [1]                     # α: 0 (zero-shot) → … → 1 (upper-bound)
#     y = [row["ub"]] + [row[c] for c in coeff_cols] + [row["zs"]]
#     plt.plot(x, y, marker="o", label=dataset_name)

# plt.xlabel("Interpolation coefficient (α)")
# plt.ylabel("Balanced Accuracy")
# plt.title("Upper-bound → Interpolated → Zero-shot balanced accuracy per dataset")
# plt.ylim(0, 1)
# plt.grid(True)
# plt.legend(fontsize="small", ncol=2)                 # tweak ncol as you like
# plt.tight_layout()
# plt.savefig("CL + Animal Trap - ML Studies Dataset with interpolation.png")
