import pandas as pd
import os
import json

csv_path = '/users/PAS2099/mino/ICICLE/other/camera-trap-CVPR - (eccv) Before CVPR Overview.csv'
output_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR-filtered-results.csv'
filtered_dataset_path = '/users/PAS2099/mino/ICICLE/plot/filtered_datasets.txt'

# 1. List of datasets visible in your screenshot
with open(filtered_dataset_path, 'r') as file:
    filtered_datasets = [l.strip() for l in file.read().splitlines() if l.strip() and not l.strip().startswith('#')]

# 2. Load the full CSV
df = pd.read_csv(csv_path)

df = df[df['dataset'] != 'dataset'] # Remove the extra header row if it exists

# 3. Filter the dataframe to only include the rows in your screenshot
df_filtered = df[df['dataset'].isin(filtered_datasets)].copy()

# 4. Fill in the best oracle data for the filtered rows
for dataset in df_filtered['dataset']:
    best_oracle_path = f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle/{dataset.replace("/", "_")}/final_training_summary.json'
    try:
        if os.path.exists(best_oracle_path):
            with open(best_oracle_path, 'r') as f:
                best_oracle_data = json.load(f)
                best_oracle_acc = best_oracle_data['averages']['balanced_accuracy']
                df_filtered.loc[df_filtered['dataset'].eq(dataset), 'best oracle'] = best_oracle_acc
        else:
            print(f"File not found for dataset: {dataset}")
    except Exception as e:
        print(f"Error processing {dataset}: {e}")

# 5. Save only the filtered version
df_filtered.to_csv(output_path, index=False)
print(f"Filtered results saved to {output_path}")