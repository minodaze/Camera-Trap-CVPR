import os
import pandas as pd

with open('/users/PAS2099/mino/ICICLE/uselist/best_accum_inter_list.txt', 'r') as f:
    datasets = [line.strip() for line in f.readlines()] 

out_path = '/users/PAS2099/mino/ICICLE/uselist/eval_best_accum_lora_model_path3.txt'

model_dir_paths = []

for dataset in datasets:
    dataset = dataset.replace("/", "_")  # Replace spaces with underscores for path construction
    json_path = f'/fs/scratch/PAS2099/camera-trap-ECCV/ascend/best_accum/{dataset}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json'
    if not os.path.isfile(json_path):
        print(f"[WARN] JSON not found for dataset {dataset}")
        json_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/{dataset}/bioclip2/lora_8_text_head/all/log/final_training_summary.json'
        if not os.path.isfile(json_path):
            print(f"[WARN] JSON not found for dataset {dataset} in fallback path")
            continue
    model_dir_path = os.path.dirname(json_path)
    model_dir_paths.append(model_dir_path)

with open(out_path, 'w') as f:
    for path in model_dir_paths:
        f.write(path + '\n')
print(f"Saved best accum model paths for {len(model_dir_paths)} datasets to {out_path}")