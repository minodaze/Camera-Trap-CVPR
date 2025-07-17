import os
from matplotlib import pyplot as plt

file_path = {
    "KGA/KGA_KHOGB03": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/lora_bsm/KGA_KGA_KHOGB03/accumulative-scratch_bsm_loss/lr_0.000025/2025-07-16-21-f9-49/bioclip2/lora_8_text_head/2025-07-12-20-6492-00/log/log.txt",
    "idaho/idaho_124": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/lora_bsm/idaho_idaho_124/accumulative-scratch_bsm_loss/lr_0.000025/2025-07-16-22-07-34/bioclip2/lora_8_text_head/2025-07-12-20-5782-00/log/log.txt",
    "na/na_archbold_FL-04": "",
    "serengeti/serengeti_L05": "",
    "na/na_archbold_FL-18": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/lora_bsm/na_na_archbold_FL-18/accumulative-scratch_bsm_loss/lr_0.000025/2025-07-16-21-57-d6/bioclip2/lora_8_text_head/2025-07-12-20-0519-00/log/log.txt",
    "na/na_archbold_FL-36": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/lora_bsm/na_na_archbold_FL-36/accumulative-scratch_bsm_loss/lr_0.000025/2025-07-16-23-7d-4f/bioclip2/lora_8_text_head/log/log.txt",
    "serengeti/serengeti_B05": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/lora_bsm/serengeti_serengeti_B05/accumulative-scratch_bsm_loss/lr_0.000025/2025-07-17-04-ff-8c/bioclip2/lora_8_text_head/log/log.txt",
    "MAD/MAD_G05": "",
    "orinoquia/orinoquia_M04": "",
    "MAD/MAD_I04": "",
    "serengeti/serengeti_G03": "",
    "serengeti/serengeti_M07": "",
    "serengeti/serengeti_J07": "",
    "serengeti/serengeti_M05": "",
    "serengeti/serengeti_Q13": "",
    "na/na_archbold_FL-41": "",
    "APN/APN_13U": "",
    "serengeti/serengeti_L09": "",
    "serengeti/serengeti_K13": "",
    "serengeti/serengeti_E08": ""
}

for k, v in file_path.items():
    assert os.path.exists(v), f"File path for {k} does not exist: {v}"
