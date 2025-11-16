import os
with open('best_accum_resume_dataset_list.txt', 'r') as file:
    datasets = file.read().splitlines()

model_paths = []
for ds in datasets:
    # ds = ds.replace("/", "_", 1)
    # /fs/scratch/PAS2099/camera-trap-final/best_accum_logs/idaho_idaho_122/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log
    # /fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_ascend/wellington_wellington_031c/bioclip2/lora_8_text_head/all/log
    model_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/{ds}/bioclip2/lora_8_text_head/all/log'
    json_path = os.path.join(model_path, 'final_training_summary.json')
    # if os.path.exists(model_path) and os.path.exists(json_path):
    if os.path.exists(model_path):
        model_paths.append(model_path)
    else:
        print(f"Model path does not exist: {model_path}")
    # else:
    #     datasets.remove(ds)

with open('best_accum_resume_model_list.txt', 'w') as file:
    for path in model_paths:
        file.write(f"{path}\n")

# with open('eval_dataset.txt', 'w') as file:
#     for ds in datasets:
#         file.write(f"{ds}\n")