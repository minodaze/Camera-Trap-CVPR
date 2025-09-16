import pandas as pd

# Read the dataset
# Replace 'your_dataset.csv' with the actual path to your file
dataset_names = [
    "orinoquia/orinoquia_N25",
    "serengeti/serengeti_E03",
    "nz/nz_EFH_HCAMD07",
    "nz/nz_EFD_DCAMF09",
    "serengeti/serengeti_D12",
    "na/na_archbold_FL-32",
    "caltech/caltech_88",
    "serengeti/serengeti_F05",
    "MAD/MAD_A05",
    "KGA/KGA_KHOGB07",
    "KGA/KGA_KHOLA03",
    "nz/nz_EFD_DCAMB02",
    "serengeti/serengeti_L09",
    "orinoquia/orinoquia_N29",
    "serengeti/serengeti_E08",
    "KGA/KGA_KHOLA08",
    "nz/nz_PS1_CAM6213",
    "nz/nz_EFH_HCAMC01",
    "nz/nz_EFH_HCAMD03",
    "nz/nz_EFH_HCAMB10"
]

import os
def dataset_to_datapath(dataset):
    """
    Convert dataset name to model path following the pattern:
    /fs/scratch/PAS2099/camera-trap-final/logs/{dataset_underscore}/upper_bound_bsm_loss/lr_0.000025/bioclip2/lora_8_text_head/log/pretrain_best_model.pth
    
    Args:
        dataset (str): Dataset name like "orinoquia/orinoquia_N25"
    
    Returns:
        str: Full model path
    """
    # Convert slashes to underscores for the path
    dataset_underscore = dataset.replace("/", "_", 1)
    # /fs/scratch/PAS2099/camera-trap-final/logs/KGA_KGA_KHOLA08/upper_bound_bsm_loss/lr_0.000025/bioclip2/lora_8_text_head/log/final_training_summary.json
    data_paths = {}
    # Build the model path
    for loss in ["bsm", "ce"]:
        for set in ["lora_8", "full"]:
            data_path = f"/fs/scratch/PAS2099/camera-trap-final/logs/{dataset_underscore}/upper_bound_{loss}_loss/lr_0.000025/bioclip2/{set}_text_head/log/final_training_summary.json"
            if not os.path.exists(data_path):
                data_path = f"/fs/scratch/PAS2099/camera-trap-final/logs/{dataset_underscore}/upper_bound_{loss}_loss/bioclip2/{set}_text_head/log/final_training_summary.json"
            data_paths[f"{set}-{loss}"] = data_path

    return data_paths

# Read the dataset
# Replace 'your_dataset.csv' with the actual path to your file
df = pd.read_csv('CL + Animal Trap - Final ML Study Dataset.csv')

import json
for dataset in dataset_names:
    data_paths = dataset_to_datapath(dataset)
    for key, path in data_paths.items():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                print(f"Loaded data from {dataset} {key}: {path}")
                # Process the JSON data here as needed
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {path}")
            continue
        balanced_accs = []
        for ckp, result in data["checkpoint_results"].items():
            balanced_accs.append(result["balanced_accuracy"])
        avg_balanced_acc = sum(balanced_accs) / len(balanced_accs) if balanced_accs else 0
        # Full FT + CE	Full FT + BSM	LoRA + CE	LoRA + BSM
        if key == "lora_8-bsm":
            df.loc[df['dataset'] == dataset, 'LoRA + BSM'] = avg_balanced_acc
        elif key == "lora_8-ce":
            df.loc[df['dataset'] == dataset, 'LoRA + CE'] = avg_balanced_acc
        elif key == "full-bsm":
            df.loc[df['dataset'] == dataset, 'Full FT + BSM'] = avg_balanced_acc
        elif key == "full-ce":
            df.loc[df['dataset'] == dataset, 'Full FT + CE'] = avg_balanced_acc
        else:
            print(f"Unknown key: {key}")

df.to_csv('CL + Animal Trap - Final ML Study Dataset_ub.csv', index=False)

