import os
with open('/users/PAS2099/mino/ICICLE/uselist/eval_dataset.txt', 'r') as file:
    datasets = file.read().splitlines()

model_paths = []
for ds in datasets:
    ds = ds.replace("/", "_", 1)
    # /fs/scratch/PAS2099/camera-trap-final/best_accum_logs/idaho_idaho_122/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_ascend/wellington_wellington_031c/bioclip2/lora_8_text_head/all/log
    model_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/{ds}/bioclip2/lora_8_text_head/all/log'
    # model_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/{ds}/bioclip2/lora_8_text_head/all/log'
    json_path = os.path.join(model_path, 'final_training_summary.json')
    # import pdb; pdb.set_trace()
    # /fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle/APN_APN_13U
    if os.path.exists(model_path) and os.path.exists(json_path):
        model_paths.append(model_path)
    else:
        model_path=f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_accum/{ds}'
        json_path = os.path.join(model_path, 'final_training_summary.json')
        if os.path.exists(model_path) and os.path.exists(json_path):
            model_paths.append(model_path)
        else:
            print(f"Warning: Model path or JSON not found for dataset {ds}")
    # else:
    #     print(f"Warning: Model path or JSON not found for dataset {ds}")
# /fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/APN_APN_U43B/bioclip2/lora_8_text_head/all/log/final_training_summary.json
with open('/users/PAS2099/mino/ICICLE/uselist/eval_model_path.txt', 'w') as file:
    for path in model_paths:
        file.write(f"{path}\n")