import os
import pandas as pd
import json

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Open-set.csv')

df["open-set best accum"] = 0
df["open-set best oracle"] = 0
df["open-set zs"] = 0

for i in range(len(df)):
    dataset = df.loc[i, "dataset"]
    dataset_path = dataset.replace("/", "_")

    best_accum_open_set_path = os.path.join(f"/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/all_class_best_accum/{dataset_path}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json")
    best_oracle_open_set_path = os.path.join(f"/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/all_class_best_oracle/{dataset_path}/bioclip2/lora_8_text_head_loss_bsm/all/log/final_training_summary.json")
    zs_open_set_path = os.path.join(f"/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/all_class_zs_3/{dataset_path}/bioclip2/full_text_head_loss_bsm/log/final_training_summary.json")
    # /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/all_class_zs_3/ENO_ENO_C04/bioclip2/full_text_head_loss_bsm/log/final_training_summary.json

    with open(best_accum_open_set_path, "r") as f:
        best_accum_open_set = json.load(f)
    with open(best_oracle_open_set_path, "r") as f:
        best_oracle_open_set = json.load(f)
    with open(zs_open_set_path, "r") as f:
        zs_open_set = json.load(f)
    df.loc[i, "open-set best accum"] = best_accum_open_set['averages']['balanced_accuracy']
    df.loc[i, "open-set best oracle"] = best_oracle_open_set['averages']['balanced_accuracy']
    df.loc[i, "open-set zs"] = zs_open_set['averages']['balanced_accuracy']

df.to_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Open-set(loaded).csv', index=False)