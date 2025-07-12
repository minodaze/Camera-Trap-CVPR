import os
from matplotlib import pyplot as plt

# os.chdir("/home/zhang.14217/bioclip-dev")
os.chdir("/users/PAS2099/mino/ICICLE")

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

def sanitize_filename(filename):
    """Sanitize filename by replacing problematic characters."""
    # Replace forward slashes and other problematic characters
    sanitized = filename.replace("/", "_").replace("\\", "_").replace(":", "-").replace(" ", "_")
    # Remove multiple consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized

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

}

avg_file_paths = {
    "channel_island_channel_island_h500ee07133326": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/channel_island_channel_island_h500ee07133326/upper_bound/2025-07-10-17-51-21/bioclip2/full_text_head/2025-07-10-17-51-27/log/log.txt",
    "idaho_idaho_261": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/idaho_idaho_261/upper_bound/2025-07-11-01-08-44/bioclip2/full_text_head/2025-07-11-01-08-50/log/log.txt",
    "KGA_KGA_KHOLA07": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/KGA_KGA_KHOLA07/upper_bound/2025-07-11-00-05-35/bioclip2/full_text_head/2025-07-11-00-05-41/log/log.txt",
    "nz_nz_Z20_3VE201": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/nz_nz_Z20_3VE201/upper_bound/2025-07-10-16-00-07/bioclip2/full_text_head/2025-07-10-16-00-28/log/log.txt",
    "serengeti_B06": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_B06/upper_bound/2025-07-12-01-05-28/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_B10": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_B10/upper_bound/2025-07-12-01-05-28/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_C05": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_C05/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_D09": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_D09/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_F01": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_F01/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_G11": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_G11/upper_bound/2025-07-12-01-05-09/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_H11": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_H11/upper_bound/2025-07-12-01-05-09/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_H13": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_H13/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_I08": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_I08/upper_bound/2025-07-12-01-05-28/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_J03": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_J03/upper_bound/2025-07-12-01-05-28/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_J09": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_J09/upper_bound/2025-07-12-01-12-47/bioclip2/full_text_head/2025-07-12-01-13-05/log/log.txt",
    "serengeti_K06": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_K06/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_L04": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_L04/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_M06": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_M06/upper_bound/2025-07-12-01-05-09/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_O08": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_O08/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_P09": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_P09/upper_bound/2025-07-12-01-05-28/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",
    "serengeti_T08": "/fs/scratch/PAS2099/mino/ICICLE/log_auto/pipeline/serengeti_serengeti_T08/upper_bound/2025-07-12-01-05-27/bioclip2/full_text_head/2025-07-12-01-05-40/log/log.txt",

}

# accu_eval_path = {
#     "ENO_C05": {
#         "accumulative" :"log/pipeline/ENO_C05_new/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-05-16-24-02/log/log.txt",
#         "regular": "log/pipeline/ENO_C05_new/ce/regular/bioclip2/full_text_head/2025-07-07-11-18-50/log/log.txt"
#     },
#     "na_lebec_CA-24": {
#         "accumulative": "log/pipeline/na_lebec_CA-24_common_name/ce/accumulative-scratch/bioclip2/full_text_head/2025-07-07-13-52-10/log/log.txt",
#         "regular": "log/pipeline/na_lebec_CA-24_common_name/ce/regular/bioclip2/full_text_head/2025-07-07-11-20-24/log/log.txt"
#     },
# }

# for dset, file_list in file_paths.items():
#     print(f"Dataset: {dset}")
#     all_acc = []
#     all_balanced_acc = []
#     all_name = []
#     for exp_name, file_path in file_list.items():
#         acc, balanced_acc = read_accu(file_path)
#         all_acc.append(acc)
#         all_balanced_acc.append(balanced_acc)
#         all_name.append(exp_name)

#     # Plot Balanced Acc
#     plt.figure(figsize=(5, 3))
    
#     # Define colors for each curve
#     colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

#     for idx, acc in enumerate(all_balanced_acc):
#         lb = all_name[idx]
#         lb = lb.replace("30:", "")
#         lb = lb.replace("cb_log:", "")
#         lb = lb.replace("b0.9", "balanced-ce")
        
#         # Use different color for each curve
#         color = colors[idx % len(colors)]
        
#         plt.plot(acc, label=lb, color=color)
#     plt.xlabel("Ckp")
#     plt.ylabel("Balanced Acc")
#     plt.title(f"{dset}")
#     plt.legend(
#         loc='center left',
#         bbox_to_anchor=(1, 0.5)
#     )
#     plt.grid()
#     plt.savefig(f"figures/{dset}balanced_acc.png", bbox_inches='tight')
#     plt.close()

# for dset, file_list in accu_eval_path.items():
#     print(f"Dataset: {dset}")
#     all_balanced_acc = []
#     all_name = []
#     zs_balanced_acc = None
#     zs_name = None
    
#     for exp_name, file_path in file_list.items():
#         if "regular" in exp_name:
#             zs_acc, zs_balanced_acc = read_accu(file_path)
#             zs_name = exp_name
#         else:
#             acc, balanced_acc = read_accu_eval(file_path)
#             all_balanced_acc.append(balanced_acc)
#             all_name.append(exp_name)

#     # Plot Balanced Acc
#     plt.figure(figsize=(5, 3))
    
#     # Define colors for each curve
#     colors = ['blue', 'lightblue', 'red', 'green', 'purple', 'brown', 'pink', 'orange', 'olive', 'cyan']
    
#     # Plot ZS baseline if it exists
#     if zs_balanced_acc is not None:
#         zs_lb = zs_name.replace("30:", "").replace("cb_log:", "").replace("b0.9", "balanced-ce")
#         plt.plot(zs_balanced_acc, label=f"{zs_lb}", color='gray', linestyle='--')
    
#     # Plot accumulative evaluation results
#     for exp_idx, balanced_acc_list in enumerate(all_balanced_acc):
#         exp_name = all_name[exp_idx]
#         exp_lb = exp_name.replace("30:", "").replace("cb_log:", "").replace("b0.9", "balanced-ce")
        
#         # Plot main curve (current checkpoint performance)
#         main_curve = [ba_list[0] for ba_list in balanced_acc_list if len(ba_list) > 0]
#         if main_curve:
#             plt.plot(main_curve, label=f"{exp_lb} (current)", color=colors[exp_idx % len(colors)], linewidth=2)
        
#         # Plot future checkpoint curves
#         for ckp_idx, ba_list in enumerate(balanced_acc_list):
#             if ckp_idx == 5:
#                 future_curve = ba_list[1:]
#                 future_x = list(range(ckp_idx + 1, ckp_idx + 1 + len(future_curve)))
#                 plt.plot(future_x, future_curve, 
#                             label=f"{exp_lb} (future from ckp_{ckp_idx})", 
#                             color=colors[exp_idx % len(colors)], 
#                             alpha=0.5, 
#                             linestyle=':')