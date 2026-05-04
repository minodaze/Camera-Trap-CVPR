import os
import pandas as pd
import json

dataset_list_path = '/users/PAS2099/mino/ICICLE/uselist/best_all_class.txt'
csv_path = 'csv/camera-trap-CVPR - ECCV (load FINAL).csv'
out_path = 'csv/camera-trap-CVPR - ECCV (load FINAL).csv'

# df = pd.read_csv(csv_path)

with open(dataset_list_path, 'r') as f:
    dataset_list = f.read().splitlines()

unfinished_dataset_path = '/users/PAS2099/mino/ICICLE/uselist/best_all_class_resume.txt'
unfinished_model_path = '/users/PAS2099/mino/ICICLE/uselist/best_all_class_resume_model_list.txt'

for dataset in dataset_list:
    dataset_path = dataset.replace('/', '_')
    folder = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/all_class_best_accum/{dataset_path}/bioclip2/lora_8_text_head_loss_bsm/all/log'
    json_path = os.path.join(folder, 'final_training_summary.json')
    if os.path.exists(json_path):
        print(f"Found final_training_summary.json for dataset: {dataset}")
        # with open(json_path, 'r') as f:
        #     summary = json.load(f)
        # # import pdb; pdb.set_trace()
        # df.loc[df['dataset'] == dataset, 'Best Accum (new)'] = summary['averages']['balanced_accuracy']
    else:
        print(f"final_training_summary.json not found for dataset: {dataset}")
        with open(unfinished_dataset_path, 'a') as f:
            f.write(f"{dataset}\n")
        with open(unfinished_model_path, 'a') as f:
            f.write(f"{folder}\n")
    
# df.to_csv(out_path, index=False)