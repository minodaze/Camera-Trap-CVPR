import os
with open('/users/PAS2099/mino/ICICLE/uselist/oracle_inter.txt', 'r') as file:
    datasets = file.read().splitlines()

oracle_model_txt_path = '/users/PAS2099/mino/ICICLE/uselist/eval_oracle_inter_model_path.txt'
bsm_oracle_model_txt_path = '/users/PAS2099/mino/ICICLE/uselist/eval_bsm_oracle_inter_model_path.txt'
lora_oracle_model_txt_path = '/users/PAS2099/mino/ICICLE/uselist/eval_lora_oracle_inter_model_path.txt'
best_oracle_model_txt_path = '/users/PAS2099/mino/ICICLE/uselist/eval_best_oracle_inter_model_path.txt'

oracle_model_paths = []
lora_oracle_model_paths = []
bsm_oracle_model_paths = []
best_oracle_model_paths = []

for ds in datasets:
    ds = ds.replace("/", "_", 1)
    oracle_model_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/oracle/{ds}/bioclip2/full_text_head/all/log'
    json_path = os.path.join(oracle_model_path, 'final_training_summary.json')
    # /fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle/APN_APN_13U
    if os.path.exists(oracle_model_path) and os.path.exists(json_path):
        oracle_model_paths.append(oracle_model_path)
    lora_oracle_model_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle/{ds}/bioclip2/lora_8_text_head/all/log'
    json_path = os.path.join(lora_oracle_model_path, 'final_training_summary.json')
    if os.path.exists(lora_oracle_model_path) and os.path.exists(json_path):
        lora_oracle_model_paths.append(lora_oracle_model_path)
    bsm_oracle_model_path = f'/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle/{ds}/bioclip2/full_text_head/all/log'
    json_path = os.path.join(bsm_oracle_model_path, 'final_training_summary.json')
    if os.path.exists(bsm_oracle_model_path) and os.path.exists(json_path):
        bsm_oracle_model_paths.append(bsm_oracle_model_path)
    best_oracle_model_path = f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle/{ds}'
    json_path = os.path.join(best_oracle_model_path, 'final_training_summary.json')
    if os.path.exists(best_oracle_model_path) and os.path.exists(json_path):
        best_oracle_model_paths.append(best_oracle_model_path)
    # else:
    #     model_path=f'/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_oracle/{ds}'
    #     json_path = os.path.join(model_path, 'final_training_summary.json')
    #     if os.path.exists(model_path) and os.path.exists(json_path):
    #         model_paths.append(model_path)
    #     else:
    #         print(f"Warning: Model path or JSON not found for dataset {ds}")

# /fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum/APN_APN_U43B/bioclip2/lora_8_text_head/all/log/final_training_summary.json
with open(oracle_model_txt_path, 'w') as file:
    for path in oracle_model_paths:
        file.write(f"{path}\n")
with open(lora_oracle_model_txt_path, 'w') as file:
    for path in lora_oracle_model_paths:
        file.write(f"{path}\n")
with open(bsm_oracle_model_txt_path, 'w') as file:
    for path in bsm_oracle_model_paths:
        file.write(f"{path}\n")
with open(best_oracle_model_txt_path, 'w') as file:
    for path in best_oracle_model_paths:
        file.write(f"{path}\n")