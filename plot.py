import os
from matplotlib import pyplot as plt

# os.chdir("/home/zhang.14217/bioclip-dev")
os.chdir("/users/PAS2099/mino/ICICLE")

def read_accu(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    acc_arr = []
    balanced_acc_arr = []
    for line in lines:
        if not "INFO - Number of samples: " in line:
            continue
        acc = line.split("acc: ")[1].split(", b")[0]
        balanced_acc = line.split("balanced acc: ")[1].split(", loss")[0]
        balanced_acc = balanced_acc.replace(". \n", "")
        # print(acc)
        acc = float(acc)
        balanced_acc = float(balanced_acc)
        acc_arr.append(acc)
        balanced_acc_arr.append(balanced_acc)
    return acc_arr, balanced_acc_arr
    # /users/PAS2099/mino/ICICLE/log/pipeline/ENO_C05_new/ce/zs/bioclip/full_text_head/2025-06-26-14-17-29/log/ckp_12_mask.pkl

file_paths = {
    "ENO_C05(bioclip-zs)":{
        "openai": "log/pipeline/ENO_C05_new/ce/zs/bioclip/full_text_head/2025-06-26-20-06-25/log/log.txt",
        "bioclip": "log/pipeline/ENO_C05_new/ce/zs/bioclip/full_text_head/2025-06-26-20-03-54/log/log.txt"
    },
    "ENO_C05(bioclip2-zs)":{
        "openai": "log/pipeline/ENO_C05_new/ce/zs/bioclip2/full_text_head/2025-06-26-21-11-58/log/log.txt",
        "bioclip": "log/pipeline/ENO_C05_new/ce/zs/bioclip2/full_text_head/2025-06-26-21-09-59/log/log.txt"
    },
    "APN_K024(bioclip-zs)":{
        "openai": "log/pipeline/APN_K024/ce/zs/bioclip/full_text_head/2025-06-26-20-10-25/log/log.txt",
        "bioclip": "log/pipeline/APN_K024/ce/zs/bioclip/full_text_head/2025-06-26-20-18-55/log/log.txt"
    },
    "APN_K024(bioclip2-zs)":{
        "openai": "log/pipeline/APN_K024/ce/zs/bioclip2/full_text_head/2025-06-26-21-04-28/log/log.txt",
        "bioclip": "log/pipeline/APN_K024/ce/zs/bioclip2/full_text_head/2025-06-26-21-08-29/log/log.txt"
    }
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
    for i, acc in enumerate(all_balanced_acc):
        lb = all_name[i]
        lb = lb.replace("30:", "")
        lb = lb.replace("cb_log:", "")
        lb = lb.replace("b0.9", "balanced-ce")
        
        # Set colors based on bioclip type
        color = None
        if 'bioclip2' in all_name[i].lower():
            color = 'orange'
        elif 'bioclip' in all_name[i].lower():
            color = 'blue'
        
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
