import json
import pandas as pd

csv_path = 'csv/camera-trap-all results(new).csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)
# Extract the 'dataset' column and convert it to a list

df['wildclip_zs'] = 0
df['wildclip_best_accum'] = 0
df['wildclip_best_oracle'] = 0

print(df.head())

dataset_path = 'uselist/wildlclip.txt'
with open(dataset_path, 'r') as f:
    datasets = [line.strip() for line in f if line.strip()]

for dataset in datasets:
    file_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/wildclip_zs/{dataset}/wildclip/full_text_head_loss_bsm/log/final_training_summary.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        best_acc = data['averages']['balanced_accuracy']
        df.loc[df['dataset'] == dataset, 'wildclip_zs'] = best_acc
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

unfinished_wildclip_accum = []
unfinished_wildclip_best_accum = []

for dataset in datasets:
    file_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/wildclip_best_accum/{dataset}/wildclip/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        best_acc = data['averages']['balanced_accuracy']
        df.loc[df['dataset'] == dataset, 'wildclip_best_accum'] = best_acc
    except Exception as e:
        unfinished_wildclip_best_accum.append(dataset)
        print(f"Error loading {file_path}: {e}")

for dataset in datasets:
    file_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/wildclip_accum/{dataset}/wildclip/full_text_head_loss_ce/all/log/final_training_summary.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        best_acc = data['averages']['balanced_accuracy']
        df.loc[df['dataset'] == dataset, 'wildclip_accum'] = best_acc
    except Exception as e:
        unfinished_wildclip_accum.append(dataset)
        print(f"Error loading {file_path}: {e}")

unfinished_wildclip_best_oracle = []
for dataset in datasets:
    file_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/best_oracle_wildclip/{dataset}/wildclip/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        best_acc = data['averages']['balanced_accuracy']
        df.loc[df['dataset'] == dataset, 'wildclip_best_oracle'] = best_acc
    except Exception as e:
        unfinished_wildclip_best_oracle.append(dataset)
        print(f"Error loading {file_path}: {e}")

output_csv_path = 'csv/camera-trap-all results(wildclip_bioclip-tem).csv'
df.to_csv(output_csv_path, index=False)

print(f"Updated results saved to {output_csv_path}")
print("Unfinished wildclip_best_accum datasets:")
for dataset in unfinished_wildclip_best_accum:
    print(f"{dataset}")
print("Unfinished wildclip_accum datasets:")
for dataset in unfinished_wildclip_accum:
    print(f"{dataset}")
print("Unfinished wildclip_best_oracle datasets:")
for dataset in unfinished_wildclip_best_oracle:
    print(f"{dataset}")