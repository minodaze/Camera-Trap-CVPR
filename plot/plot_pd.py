import pandas as pd
import matplotlib.pyplot as plt
import json

method = 'kms'

with open('plot/plot.txt', 'r') as file:
    dataset = file.read().splitlines()

columns = ['dataset', 'ratio-0.1', 'ratio-0.3', 'ratio-0.5', 'ratio-0.7', 'ratio-0.9']

print(dataset)

# df = pd.DataFrame([line.split() for line in data], columns=columns)

data = {
    'dataset': dataset,
    'zs': [],
    'ratio-0.1': [],
    'ratio-0.2': [],
    'ratio-0.3': [],
    'ratio-0.4': [],
    'ratio-0.5': [],
    'ratio-0.6': [],
    'ratio-0.7': [],
    'ratio-0.8': [],
    'ratio-0.9': [],
    'all-ub': [],
    'max': [],
    'max_column': []
}

ratios = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

for ratio in ratios:
    for ds in dataset:
        ds = ds.replace('/', '_', 1)
        json_path_template1 = f'/fs/scratch/PAS2099/camera-trap-final/AL_logs/{ds}/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/{method}_percentage_{ratio}_num_samples_per_cluster_1/log/final_training_summary.json'
        json_path_template2 = f'/fs/scratch/PAS2099/camera-trap-final/AL_logs/{ds}/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/{method}_percentage_{ratio}/log/final_training_summary.json'
        try:
            with open(json_path_template1, 'r') as f:
                json_data = json.load(f)
                balanced_acc = json_data['averages']['balanced_accuracy']
                data[f'ratio-{ratio}'].append(balanced_acc)
        except FileNotFoundError:
            print(f"File not found: {json_path_template1}, trying {json_path_template2}")
            try:
                with open(json_path_template2, 'r') as f:
                    json_data = json.load(f)
                    balanced_acc = json_data['averages']['balanced_accuracy']
                    data[f'ratio-{ratio}'].append(balanced_acc)
            except FileNotFoundError:
                print(f"File not found: {json_path_template2}")
                data[f'ratio-{ratio}'].append(None)

for ds in dataset:
    ds = ds.replace('/', '_', 1)
    json_path_ub = f'/fs/scratch/PAS2099/camera-trap-final/AL_logs/{ds}/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/log/final_training_summary.json'
    json_path_zs = f'/fs/scratch/PAS2099/camera-trap-final/AL_logs/{ds}/zs_ce_loss/bioclip2/full_text_head/log/final_training_summary.json'
    try:
        with open(json_path_ub, 'r') as f:
            json_data_ub = json.load(f)
            balanced_acc_ub = json_data_ub['averages']['balanced_accuracy']
            data['all-ub'].append(balanced_acc_ub)
    except FileNotFoundError:
        print(f"File not found: {json_path_ub}")
        data['all-ub'].append(None)
    try:
        with open(json_path_zs, 'r') as f:
            json_data_zs = json.load(f)
            balanced_acc_zs = json_data_zs['averages']['balanced_accuracy']
            data['zs'].append(balanced_acc_zs)
    except FileNotFoundError:
        print(f"File not found: {json_path_zs}")
        data['zs'].append(None)

data['max'] = [max(filter(None, [data[f'ratio-{r}'][i] for r in ratios] + [data['all-ub'][i], data['zs'][i]])) for i in range(len(dataset))]
data['max_column'] = []
for i in range(len(dataset)):
    values = {}
    for r in ratios:
        if data[f'ratio-{r}'][i] is not None:
            values[f'ratio-{r}'] = data[f'ratio-{r}'][i]
    if data['all-ub'][i] is not None:
        values['all-ub'] = data['all-ub'][i]
    if data['zs'][i] is not None:
        values['zs'] = data['zs'][i]
    
    max_column = max(values, key=values.get) if values else None
    data['max_column'].append(max_column)

df = pd.DataFrame(data)
df.to_csv(f'csv/{method}.csv', index=False)
