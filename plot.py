import os
from matplotlib import pyplot as plt

# os.chdir("/home/zhang.14217/bioclip-dev")
os.chdir("/users/PAS2099/mino/ICICLE")

def read_accu(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    acc_arr = []
    balanced_acc_arr = []
    ckp_idx = 1
    for line in lines:
        if "INFO - Number of samples: " in line or f"INFO - Accu-eval start on ckp_{ckp_idx} at ckp_{ckp_idx}:" in line:
            acc = line.split("acc: ")[1].split(", b")[0]
            balanced_acc = line.split("balanced acc: ")[1].split(", loss")[0]
            balanced_acc = balanced_acc.replace(". \n", "")
            # print(acc)
            acc = float(acc)
            balanced_acc = float(balanced_acc)
            acc_arr.append(acc)
            balanced_acc_arr.append(balanced_acc)
            ckp_idx += 1
    return acc_arr, balanced_acc_arr
    # /users/PAS2099/mino/ICICLE/log/pipeline/ENO_C05_new/ce/zs/bioclip/full_text_head/2025-06-26-14-17-29/log/ckp_12_mask.pkl

def read_accu_eval(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Dictionary to store results for each checkpoint
    checkpoint_results = {}
    
    for line in lines:
        if "Accu-eval start on ckp_" in line and " at ckp_" in line:
            try:
                parts = line.split("Accu-eval start on ckp_")[1]
                model_ckp = int(parts.split(" at ckp_")[0])
                test_ckp = int(parts.split(" at ckp_")[1].split(":")[0])
                
                acc = float(line.split("acc: ")[1].split(", b")[0])
                balanced_acc = float(line.split("balanced acc: ")[1].split(", loss")[0].replace(". \n", ""))
                
                if model_ckp not in checkpoint_results:
                    checkpoint_results[model_ckp] = {'acc': [], 'balanced_acc': [], 'test_ckps': []}
                
                checkpoint_results[model_ckp]['acc'].append(acc)
                checkpoint_results[model_ckp]['balanced_acc'].append(balanced_acc)
                checkpoint_results[model_ckp]['test_ckps'].append(test_ckp)
                
            except (ValueError, IndexError) as e:
                continue
    
    # Convert to lists ordered by checkpoint number
    acc_arr = []
    balanced_acc_arr = []
    
    for ckp in sorted(checkpoint_results.keys()):
        acc_arr.append(checkpoint_results[ckp]['acc'])
        balanced_acc_arr.append(checkpoint_results[ckp]['balanced_acc'])
    
    return acc_arr, balanced_acc_arr

file_paths = {
    "ENO_C05-common name": {
        "zs": "log/pipeline/ENO_C05_new/ce/zs/bioclip2/full_text_full/2025-07-02-22-29-13/log/log.txt",
        "accumulative(lora-vision+text)": "log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/lora_8_text_lora/2025-07-03-13-43-25/log/log.txt",
        "accumulative(lora-vision+head)": "log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/lora_8_text_head/2025-07-03-13-43-43/log/log.txt",
        "accumulative(vision+text)": "log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/full_text_full/2025-07-02-22-26-22/log/log.txt",
        "accumulative(vision+head)": "log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-02-23-39-01/log/log.txt"
    },
    "na_lebec_CA-24-common name": {
        "zs": "log/pipeline/na_lebec_CA-24_common_name/ce/zs/bioclip2/full_text_head/2025-07-02-23-32-26/log/log.txt",
        "accumulative(lora-vision+text)": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/lora_8_text_lora/2025-07-03-12-49-58/log/log.txt",
        "accumulative(lora-vision+head)": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/lora_8_text_head/2025-07-03-12-53-50/log/log.txt",
        "accumulative(vision+text)": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/full_text_full/2025-07-07-14-55-33/log/log.txt",
        "accumulative(vision+head)": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-02-23-30-18/log/log.txt"
    },
    "ENO_C05-common name-regularization(vision+head)": {
        "zs": "log/pipeline/ENO_C05_new/ce/zs/bioclip2/full_text_full/2025-07-02-22-29-13/log/log.txt",
        "accumulative": "log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-05-16-24-02/log/log.txt",
        "accumulative-lwf": "log/pipeline/ENO_C05_new/kd/accumulative-scratch-lwf/bioclip2/full_text_head/2025-07-05-23-39-35/log/log.txt",
        "accumulative-interpolation": "log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-05-18-00-45/log/log.txt",
        "CLEAR-lwf": "log/pipeline/ENO_C05_new/kd/lwf/bioclip2/full_text_head/2025-07-05-23-39-06/log/log.txt",
        "CLEAR-interpolation": "log/pipeline/ENO_C05_new/ce/replay/bioclip2/full_text_head_interpolation_model_0.5/2025-07-07-13-33-48/log/log.txt",
    },
    "na_lebec_CA-24_common_name-regularization-vision+head": {
        "zs": "log/pipeline/na_lebec_CA-24_common_name/ce/zs/bioclip2/full_text_head/2025-07-02-23-32-26/log/log.txt",
        "accumulative": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-02-23-30-18/log/log.txt",
        "accumulative-lwf": "log/pipeline/na_lebec_CA-24_common_name/kd/accumulative-scratchLWF/bioclip2/full_text_head/2025-07-06-15-54-27/log/log.txt",
        "accumulative-interpolation": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/full_text_head_interpolation_model_0.5/2025-07-06-15-35-58/log/log.txt",
        "CLEAR-lwf": "log/pipeline/na_lebec_CA-24_common_name/kd/lwf/bioclip2/full_text_head/2025-07-07-13-45-10/log/log.txt",
        "CLEAR-interpolation": "log/pipeline/na_lebec_CA-24_common_name/ce/replay/bioclip2/full_text_head_interpolation_model_0.5/2025-07-07-13-44-48/log/log.txt"
    },
    "ENO_C05-common name-zs": {
        "petl": "log/pipeline/ENO_C05_new/ce/zs/bioclip2/full_text_head/log.txt",
        "openclip": "/users/PAS2099/mino/ICICLE/log/pipeline/ENO_C05_new/ce/zs/bioclip2/full_text_head/2025-07-02-23-33-28/log/log.txt"
    },
}
accu_eval_path = {
    "ENO_C05": {
        "accumulative" :"log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-05-16-24-02/log/log.txt",
        "regular": "log/pipeline/ENO_C05_new/ce/regular/bioclip2/full_text_head/2025-07-07-11-18-50/log/log.txt"
    },
    "na_lebec_CA-24": {
        "accumulative": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-07-13-52-10/log/log.txt",
        "regular": "log/pipeline/na_lebec_CA-24_common_name/ce/regular/bioclip2/full_text_head/2025-07-07-11-20-24/log/log.txt"
    },
}

for dset, file_list in file_paths.items():
    print(f"Dataset: {dset}")
    all_acc = []
    all_balanced_acc = []
    all_name = []
    for exp_name, file_path in file_list.items():
        acc, balanced_acc = read_accu(file_path)
        all_acc.append(acc)
        all_balanced_acc.append(balanced_acc)
        all_name.append(exp_name)

    # Plot Balanced Acc
    plt.figure(figsize=(5, 3))
    
    # Define colors for each curve
    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for idx, acc in enumerate(all_balanced_acc):
        lb = all_name[idx]
        lb = lb.replace("30:", "")
        lb = lb.replace("cb_log:", "")
        lb = lb.replace("b0.9", "balanced-ce")
        
        # Use different color for each curve
        color = colors[idx % len(colors)]
        
        plt.plot(acc, label=lb, color=color)
    plt.xlabel("Ckp")
    plt.ylabel("Balanced Acc")
    plt.title(f"{dset}")
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    plt.grid()
    plt.savefig(f"figures/{dset}balanced_acc.png", bbox_inches='tight')
    plt.close()

for dset, file_list in accu_eval_path.items():
    print(f"Dataset: {dset}")
    all_balanced_acc = []
    all_name = []
    zs_balanced_acc = None
    zs_name = None
    
    for exp_name, file_path in file_list.items():
        if "regular" in exp_name:
            zs_acc, zs_balanced_acc = read_accu(file_path)
            zs_name = exp_name
        else:
            acc, balanced_acc = read_accu_eval(file_path)
            all_balanced_acc.append(balanced_acc)
            all_name.append(exp_name)

    # Plot Balanced Acc
    plt.figure(figsize=(5, 3))
    
    # Define colors for each curve
    colors = ['blue', 'lightblue', 'red', 'green', 'purple', 'brown', 'pink', 'orange', 'olive', 'cyan']
    
    # Plot ZS baseline if it exists
    if zs_balanced_acc is not None:
        zs_lb = zs_name.replace("30:", "").replace("cb_log:", "").replace("b0.9", "balanced-ce")
        plt.plot(zs_balanced_acc, label=f"{zs_lb}", color='gray', linestyle='--')
    
    # Plot accumulative evaluation results
    for exp_idx, balanced_acc_list in enumerate(all_balanced_acc):
        exp_name = all_name[exp_idx]
        exp_lb = exp_name.replace("30:", "").replace("cb_log:", "").replace("b0.9", "balanced-ce")
        
        # Plot main curve (current checkpoint performance)
        main_curve = [ba_list[0] for ba_list in balanced_acc_list if len(ba_list) > 0]
        if main_curve:
            plt.plot(main_curve, label=f"{exp_lb} (current)", color=colors[exp_idx % len(colors)], linewidth=2)
        
        # Plot future checkpoint curves
        for ckp_idx, ba_list in enumerate(balanced_acc_list):
            if ckp_idx == 5:
                future_curve = ba_list[1:]
                future_x = list(range(ckp_idx + 1, ckp_idx + 1 + len(future_curve)))
                plt.plot(future_x, future_curve, 
                            label=f"{exp_lb} (future from ckp_{ckp_idx})", 
                            color=colors[exp_idx % len(colors)], 
                            alpha=0.5, 
                            linestyle=':')

    plt.xlabel("Ckp")
    plt.ylabel("Balanced Acc")
    plt.title(f"{dset}")
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    plt.grid()
    plt.savefig(f"figures/{dset}balanced_acc.png", bbox_inches='tight')
    plt.close()