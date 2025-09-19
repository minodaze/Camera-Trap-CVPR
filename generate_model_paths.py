with open('eval_dataset.txt', 'r') as file:
    datasets = file.read().splitlines()

model_paths = []
for ds in datasets:
    ds_name = ds.replace("/", "_", 1)
    # /fs/scratch/PAS2099/camera-trap-final/best_accum_logs/idaho_idaho_122/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log
    model_path = f'/fs/scratch/PAS2099/camera-trap-final/best_accum_logs/{ds_name}/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log'
    model_paths.append(model_path)

with open('model_path.txt', 'w') as file:
    for path in model_paths:
        file.write(f"{path}\n")