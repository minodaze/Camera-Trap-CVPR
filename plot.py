# import os
# from matplotlib import pyplot as plt

# # os.chdir("/home/zhang.14217/bioclip-dev")
# # os.chdir("/users/PAS2099/mino/ICICLE")

# def read_accu(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     acc_arr = []
#     balanced_acc_arr = []
#     ckp_idx = 1
#     for line in lines:
#         if "INFO - Number of samples: " in line or f"INFO - Accu-eval start on ckp_{ckp_idx} at ckp_{ckp_idx}:" in line:
#             acc = line.split("acc: ")[1].split(", b")[0]
#             balanced_acc = line.split("balanced acc: ")[1].split(", loss")[0]
#             balanced_acc = balanced_acc.replace(". \n", "")
#             # print(acc)
#             acc = float(acc)
#             balanced_acc = float(balanced_acc)
#             acc_arr.append(acc)
#             balanced_acc_arr.append(balanced_acc)
#             ckp_idx += 1
#     return acc_arr, balanced_acc_arr
#     # /users/PAS2099/mino/ICICLE/log/pipeline/ENO_C05_new/ce/zs/bioclip/full_text_head/2025-06-26-14-17-29/log/ckp_12_mask.pkl

# def read_accu_eval(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # Dictionary to store results for each checkpoint
#     checkpoint_results = {}
    
#     for line in lines:
#         if "Accu-eval start on ckp_" in line and " at ckp_" in line:
#             try:
#                 parts = line.split("Accu-eval start on ckp_")[1]
#                 model_ckp = int(parts.split(" at ckp_")[0])
#                 test_ckp = int(parts.split(" at ckp_")[1].split(":")[0])
                
#                 acc = float(line.split("acc: ")[1].split(", b")[0])
#                 balanced_acc = float(line.split("balanced acc: ")[1].split(", loss")[0].replace(". \n", ""))
                
#                 if model_ckp not in checkpoint_results:
#                     checkpoint_results[model_ckp] = {'acc': [], 'balanced_acc': [], 'test_ckps': []}
                
#                 checkpoint_results[model_ckp]['acc'].append(acc)
#                 checkpoint_results[model_ckp]['balanced_acc'].append(balanced_acc)
#                 checkpoint_results[model_ckp]['test_ckps'].append(test_ckp)
                
#             except (ValueError, IndexError) as e:
#                 continue
    
#     # Convert to lists ordered by checkpoint number
#     acc_arr = []
#     balanced_acc_arr = []
    
#     for ckp in sorted(checkpoint_results.keys()):
#         acc_arr.append(checkpoint_results[ckp]['acc'])
#         balanced_acc_arr.append(checkpoint_results[ckp]['balanced_acc'])
    
#     return acc_arr, balanced_acc_arr

# def plot_experiment_group(group_name, experiments, title):
#     """Plot grouped experiments for comparison"""
#     plt.figure(figsize=(12, 8))
#     colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
#     for i, (exp_name, file_path) in enumerate(experiments.items()):
#         try:
#             if os.path.exists(file_path):
#                 acc, balanced_acc = read_accu(file_path)
#                 if balanced_acc:  # Only plot if we have data
#                     color = colors[i % len(colors)]
#                     plt.plot(balanced_acc, label=exp_name, color=color, linewidth=2)
#                     print(f"  {exp_name}: {len(balanced_acc)} data points")
#                 else:
#                     print(f"  {exp_name}: No data found in log file")
#             else:
#                 print(f"  {exp_name}: Log file not found at {file_path}")
#         except Exception as e:
#             print(f"  {exp_name}: Error reading log file - {e}")
    
#     plt.xlabel("Checkpoint")
#     plt.ylabel("Balanced Accuracy")
#     plt.title(title)
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.grid(True, alpha=0.3)
    
#     # Create figures directory if it doesn't exist
#     os.makedirs("figures", exist_ok=True)
    
#     plt.savefig(f"figures/{group_name}_comparison.png", bbox_inches='tight', dpi=150)
#     plt.close()
#     print(f"Saved plot: figures/{group_name}_comparison.png")

# file_paths = {

#     #  "ENO_B06(full finetune)": {
#     #     "Accmu": "log/pipeline/ENO_B06/ce/accumulative-scratch/bioclip2/full/log/log.txt",
#     #     "Full-Training": "log/pipeline/ENO_B06/ce/percentage-1/bioclip2/full/log/log.txt",
#     #     "ZS": "log/pipeline/ENO_B06/ce/zs/bioclip2/full/log/log.txt",
#     #     "Regular": "log/pipeline/ENO_B06/ce/regular/bioclip2/full/log/log.txt",
#     #     "Replay": "log/pipeline/ENO_B06/ce/CLReplay/bioclip2/full/log/log.txt",
#     #     "LWF": "log/pipeline/ENO_B06/ce/LWF/bioclip2/full/log/log.txt",
#     #     "MIR": "log/pipeline/ENO_B06/ce/mir/bioclip2/full/log/log.txt",
#     #     "RandReplaceOld": "log/pipeline/ENO_B06/ce/RandReplaceOld/bioclip2/full/log/log.txt"
#     # },
#     # "ENO_B06(petl-lora 8)": {
#     #     "Accmu": "log/pipeline/ENO_B06/ce/accumulative-scratch/bioclip2/lora_8/log/log.txt",
#     #     "Full-Training": "log/pipeline/ENO_B06/ce/percentage-1/bioclip2/lora_8/log/log.txt",
#     #     "ZS": "log/pipeline/ENO_B06/ce/zs/bioclip2/lora_8/log/log.txt",
#     #     "Regular": "log/pipeline/ENO_B06/ce/regular/bioclip2/lora_8/log/log.txt",
#     #     "Replay": "log/pipeline/ENO_B06/ce/CLReplay/bioclip2/lora_8/log/log.txt",
#     #     "LWF": "log/pipeline/ENO_B06/ce/LWF/bioclip2/lora_8/log/log.txt",
#     #     "MIR": "log/pipeline/ENO_B06/ce/mir/bioclip2/lora_8/log/log.txt",
#     #     "RandReplaceOld": "log/pipeline/ENO_B06/ce/RandReplaceOld/bioclip2/lora_8/log/log.txt"
#     # },
#     # "ENO_E06": {
#     #     "zs_sci": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_E06/ce/zs/bioclip2_2025-06-30-22-22-23/full/log/log.txt",
#     #     "zs_cmm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_E06/ce/zs/bioclip2_2025-06-30-22-22-26_common_name/full/log/log.txt"
#     # }
# #     "na_lebec_CA-22": {
# #         "zs": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log__/pipeline/na_lebec_CA-22/ce/zs/bioclip2_2025-06-27-12-10-33/full/log/log.txt",
# #         "accumulative": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log__/pipeline/na_lebec_CA-22/ce/accumulative-scratch/bioclip2_2025-06-27-01-15-39/full/log/log.txt",
# #         "upper-bound": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log__/pipeline/na_lebec_CA-22/ce/percentage-1/bioclip2_2025-06-27-11-38-06/full/log/log.txt"
# # }
# #     "na_lebec_CA-22": {
# #         "zs": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/zs/bioclip2_2025-06-27-23-17-06_common_name/full/log/log.txt",
# #         "accumulative": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/accumulative-scratch/bioclip2_2025-06-27-23-04-46_common_name/full/log/log.txt",
# #         "upper-bound": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/percentage-1/bioclip2_2025-06-27-23-15-25_common_name/full/log/log.txt"
# # }
# #     "na_lebec_CA-22": {
# #         "zs": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/zs/bioclip2_2025-06-27-23-17-06_common_name/full/log/log.txt",
# #         "accumulative(full)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/wandb/wandb/run-20250630_125319-oh5iu7ag/files/output.log",
# #         "accumulative(lora)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/cdt/accumulative-scratch/bioclip2_2025-06-30-23-06-27_common_name/lora_8/log/log.txt",
# #         "upper-bound(full)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/percentage-1/bioclip2_2025-06-30-12-56-18_common_name/full/log/log.txt",
# #         "upper-bound(lora)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/cdt/percentage-1/bioclip2_2025-06-30-23-06-27_common_name/lora_8/log/log.txt"

# # }

# #     "na_lebec_CA-22": {
# #         "zs_comm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/zs/bioclip2_2025-06-27-23-17-06_common_name/full/log/log.txt",
# #         "zs_sci": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log__/pipeline/na_lebec_CA-22/ce/zs/bioclip2_2025-06-27-12-10-33/full/log/log.txt",
# #         "acc_comm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/accumulative-scratch/bioclip2_2025-06-27-23-04-46_common_name/full/log/log.txt",
# #         "acc_sci": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log__/pipeline/na_lebec_CA-22/ce/accumulative-scratch/bioclip2_2025-06-27-01-15-39/full/log/log.txt"
# # }
# #     "APN_K024": {
# #         "zs_cmm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/zs/bioclip2_2025-06-30-12-38-49_common_name/full/log/log.txt",
# #         "upper_cmm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-06-30-00-01-23_common_name/full/log/log.txt",
# #         "acc_cmm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/accumulative-scratch/bioclip2_2025-06-29-23-57-16_common_name/full/log/log.txt",

# # }
# #     "APN_K024": {
# #         "zs_query_asc_setting": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/zs/bioclip2_2025-06-30-12-54-23_common_name/full/log/log.txt",
# # }

# # na_lebec_CA-22, smm_setting, focal, full and lora
# #     "na_lebec_CA-22": {
# #         "zs": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/zs/bioclip2_2025-06-27-23-17-06_common_name/full/log/log.txt",
# #         "accumulative(full)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/cb-focal/accumulative-scratch/bioclip2_2025-07-01-00-35-24_common_name/full/log/log.txt",
# #         "upper-bound(full)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/cb-focal/percentage-1/bioclip2_2025-07-01-00-02-12_common_name/full/log/log.txt",
# #         "accumulative(lora)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/cb-focal/accumulative-scratch/bioclip2_2025-07-01-00-42-28_common_name/lora_8/log/log.txt",
# #         "upper-bound(lora)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/cb-focal/percentage-1/bioclip2_2025-07-01-00-43-16_common_name/lora_8/log/log.txt"
        
# # }
#     # "ENO_C05":{
#     #     # "zs(scientific-name)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/ENO_C05_bioclip2_zs_log.txt",
#     #     # "zs(common-name)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/zs/bioclip2_2025-07-01-11-18-48_common_name/full/log/log.txt",
#     #      "accumulative(sci)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/ENO_C05_bioclip2_accu_log.txt",
#     #     "accumulative(cmm)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt"
#     #     # "upper-bound": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt"
#     # }

#     # "serengeti_C02": {
#     #     "accumulative(full)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/serengeti_C02/ce/accumulative-scratch/bioclip2_2025-07-01-01-11-18_common_name/full/log/log.txt",
#     #     "accumulative(lora)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/accumulative-scratch/bioclip2_2025-07-01-00-20-15_common_name/lora_8/log/log.txt",
#     #     "upper-bound(full)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-06-30-23-06-27_common_name/full/log/log.txt",
#     #     "upper-bound(lora)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-01-00-20-51_common_name/lora_8/log/log.txt",
#     #     "zs": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/zs/bioclip2_2025-06-30-12-38-49_common_name/full/log/log.txt"
#     # }

#     # "APN_K024": {
#     #     "zs(query)": "/fs/scratch/PAS2099/Lemeng/icicle/log/pipeline/APN_K024/zs/bioclip2/bioclip2/linear/log/log.txt",
#     #     "zs(cmm)": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/APN_K024/ce/zs/bioclip2_2025-06-30-12-38-49_common_name/full/log/log.txt"
#     # }

#     # "ENO_C05": {
#     #     "bioclip(common)": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/ENO_C05/ce/zs/bioclip_2025-07-03-11-46-36_common_name/full/log/log.txt",
#     #     "bioclip(query)": "/fs/scratch/PAS2099/Lemeng/icicle/log/pipeline/ENO_C05/zs/bioclip1/bioclip/full/log/log.txt",
#     #     "bioclip(scientific)": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/ENO_C05/ce/zs/bioclip_2025-07-03-11-48-18_scientific_name/full/log/log.txt",
#     #     "bioclip2(common)": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/ENO_C05/ce/zs/bioclip2_2025-07-02-16-25-01_common_name/full/log/log.txt",
#     #     "bioclip2(query)": "/fs/scratch/PAS2099/Lemeng/icicle/log/pipeline/ENO_C05/zs/bioclip2/bioclip2/linear/log/log.txt",
#     #     "bioclip2(scientific)": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/ENO_C05/ce/zs/bioclip2_2025-07-03-11-24-07_scientific_name/full/log/log.txt"
#     # },

#     # "na_CA24": {
#     #     "bioclip(common)": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/na_lebec_CA-24/ce/zs/bioclip_2025-07-03-11-47-08_common_name/full/log/log.txt",
#     #     "bioclip(query)": "/fs/scratch/PAS2099/Lemeng/icicle/log/pipeline/na_lebec_CA-24/zs/bioclip1/bioclip/full/log/log.txt",
#     #     "bioclip2(common)": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/na_lebec_CA-24/ce/zs/bioclip2_2025-07-03-11-32-40_common_name/full/log/log.txt",
#     #     "bioclip2(query)": "/fs/scratch/PAS2099/Lemeng/icicle/log/pipeline/na_lebec_CA-24/zs/bioclip2/bioclip2/linear/log/log.txt"
#     # }

#     # "test":{
#     #     "1": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/na_lebec_CA-24/ce/zs/bioclip_2025-07-03-15-35-18_common_name/full/log/log.txt",
#     #     "2": "/fs/scratch/PAS2099/Jiacheng/log/pipeline/na_lebec_CA-24/ce/zs/bioclip2_2025-07-03-15-31-46_common_name/full/log/log.txt"
#     # }

#     # "ENO_C05_common": {
#     #     "acc(ce)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt",
#     #     "acc(cdt_0.05)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/accumulative-scratch/bioclip2_2025-07-03-00-49-54_common_name/full/log/log.txt",
#     #     "acc(cdt_0.1)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/accumulative-scratch/bioclip2_2025-07-03-00-37-41_common_name/full/log/log.txt",
#     #     "acc(cdt_0.2)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/accumulative-scratch/bioclip2_2025-07-03-00-40-53_common_name/full/log/log.txt",
#     #     "acc(cdt_0.3)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/accumulative-scratch/bioclip2_2025-07-02-23-28-37_common_name/full/log/log.txt",
#     #     "acc(cdt_0.4)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/accumulative-scratch/bioclip2_2025-07-03-00-42-41_common_name/full/log/log.txt",
#     #     "acc(cdt_0.5)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/accumulative-scratch/bioclip2_2025-07-03-00-44-21_common_name/full/log/log.txt"
    
#     # }

#     #     "ENO_C05_common": {
#     #     "upper_bound(ce)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.05)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-01-10-25_common_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.1)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-00-51-47_common_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.2)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-00-54-09_common_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.3)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-02-23-29-06_common_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.4)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-00-55-34_common_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.5)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-01-09-09_common_name/full/log/log.txt"

#     # }

#     #     "APN_K024_common": {
#     #     "acc(ce)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/accumulative-scratch/bioclip2_2025-06-29-23-57-16_common_name/full/log/log.txt",
#     #     "acc(cdt_0.05)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/accumulative-scratch/bioclip2_2025-07-03-01-28-24_common_name/full/log/log.txt",
#     #     "acc(cdt_0.1)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/accumulative-scratch/bioclip2_2025-07-03-01-25-43_common_name/full/log/log.txt",
#     #     "acc(cdt_0.2)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/accumulative-scratch/bioclip2_2025-07-03-01-26-51_common_name/full/log/log.txt",
#     #     "acc(cdt_0.3)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/accumulative-scratch/bioclip2_2025-06-30-22-47-59_common_name/full/log/log.txt",
#     #     "acc(cdt_0.4)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/accumulative-scratch/bioclip2_2025-07-03-01-27-14_common_name/full/log/log.txt",
#     #     "acc(cdt_0.5)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/accumulative-scratch/bioclip2_2025-07-03-01-27-44_common_name/full/log/log.txt"
    
#     # }
#     #         "ENO_C05_scientific": {
#     #     "upper_bound(ce)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-07-05-11-57-07_scientific_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.05)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-06-22-34-24_scientific_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.1)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-06-22-38-24_scientific_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.2)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-06-22-39-51_scientific_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.3)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-06-22-41-54_scientific_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.4)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-06-22-44-24_scientific_name/full/log/log.txt",
#     #     "upper_bound(cdt_0.5)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-06-22-45-54_scientific_name/full/log/log.txt"

#     # }

#     # "na_lebec_CA-24_common": {
#     #     "acc(ce)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/ce/accumulative-scratch/bioclip2_2025-07-03-01-03-32_common_name/full/log/log.txt",
#     #     "acc(cdt_0.05)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-19-19_common_name/full/log/log.txt",
#     #     "acc(cdt_0.1)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-15-03_common_name/full/log/log.txt",
#     #     "acc(cdt_0.2)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-17-00_common_name/full/log/log.txt",
#     #     "acc(cdt_0.3)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-06-47_common_name/full/log/log.txt",
#     #     "acc(cdt_0.4)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-17-52_common_name/full/log/log.txt",
#     #     "acc(cdt_0.5)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-18-31_common_name/full/log/log.txt" 
#     # }

#     # "ENO_C05_common_focal":{
#     #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt",
#     #     "gamma_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-24-21_common_name/full/log/log.txt",
#     #     "gamma_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-26-19_common_name/full/log/log.txt",
#     #     "gamma_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-27-27_common_name/full/log/log.txt",
#     #     "gamma_1.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-28-44_common_name/full/log/log.txt",
#     #     "gamma_2.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-32-02_common_name/full/log/log.txt",
#     #     "gamma_5.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-32-55_common_name/full/log/log.txt"
#     # }

# #     "ENO_C05_common_focal":{
# #         "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
# #         "gamma_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-36-29_common_name/full/log/log.txt",        
# #         "gamma_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-37-59_common_name/full/log/log.txt",    
# #         "gamma_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-39-31_common_name/full/log/log.txt",    
# #         "gamma_1.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-40-30_common_name/full/log/log.txt",    
# #         "gamma_2.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-43-34_common_name/full/log/log.txt",    
# #         "gamma_5.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-44-36_common_name/full/log/log.txt" 


# # }

#         #   "ENO_C05_common_focal":{
#         #       "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt",
#         #       "dynamic_beta": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-ce/accumulative-scratch/bioclip2_2025-07-07-21-16-08_common_name/full/log/log.txt",
#         #       "beta_0.99": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-ce/accumulative-scratch/bioclip2_2025-07-07-19-46-44_common_name/full/log/log.txt"
#         #   }

#         # "ENO_C05_common_cb_focal":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt",
#         #     "dynamic_beta": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/accumulative-scratch/bioclip2_2025-07-07-19-21-37_common_name/full/log/log.txt",
#         #     "beta_0.9": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/accumulative-scratch/bioclip2_2025-07-07-19-24-57_common_name/full/log/log.txt",
#         #     "beta_0.99": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/accumulative-scratch/bioclip2_2025-07-07-19-25-24_common_name/full/log/log.txt",
#         #     "beta_0.999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/accumulative-scratch/bioclip2_2025-07-07-19-26-29_common_name/full/log/log.txt",
#         #     "beta_0.9999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/accumulative-scratch/bioclip2_2025-07-07-19-27-24_common_name/full/log/log.txt"


#         # }

#         # "ENO_C05_common_cb_focal":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
#         #     "dynamic_beta": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/percentage-1/bioclip2_2025-07-07-19-29-28_common_name/full/log/log.txt",
#         #     "beta_0.9": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/percentage-1/bioclip2_2025-07-07-19-30-58_common_name/full/log/log.txt",
#         #     "beta_0.99": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/percentage-1/bioclip2_2025-07-07-19-31-58_common_name/full/log/log.txt",
#         #     "beta_0.999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/percentage-1/bioclip2_2025-07-07-19-32-58_common_name/full/log/log.txt",
#         #     "beta_0.9999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cb-focal/percentage-1/bioclip2_2025-07-07-19-33-53_common_name/full/log/log.txt"
#         # }

#         # "ENO_C05_common_LDAM":{
#         #     "CE":"/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
#         #     "LDAM": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ldam/percentage-1_new_settings/bioclip2_2025-07-07-20-12-29_common_name/full/log/log.txt"
#         # }

#         #         "ENO_C05_common_LDAM":{
#         #     "CE":"/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt",
#         #     "LDAM": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ldam/accumulative-scratch/bioclip2_2025-07-07-20-10-24_common_name/full/log/log.txt"
#         # }

#         # "ENO_C05_common_bsm":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt",
#         #     "bsm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/bsm/accumulative-scratch/bioclip2_2025-07-07-20-17-30_common_name/full/log/log.txt"
#         # }

#         # "ENO_C05_common_bsm":{
#         #     "CE":"/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
#         #     "bsm": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/bsm/percentage-1/bioclip2_2025-07-07-20-20-55_common_name/full/log/log.txt"
#         # }

#         # "APN_K024_common_cdt":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-06-30-00-01-23_common_name/full/log/log.txt",
#         #     "cdt_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-03-01-26-26_common_name/full/log/log.txt",
#         #     "cdt_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-03-01-27-02_common_name/full/log/log.txt",
#         #     "cdt_0.3":"/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-06-30-23-06-27_common_name/full/log/log.txt",
#         #     "cdt_0.4": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-03-01-27-30_common_name/full/log/log.txt",
#         #     "cdt_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-03-01-27-56_common_name/full/log/log.txt",
#         #     "cdt_0.05": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-03-01-28-49_common_name/full/log/log.txt"
#         # }

#         # "APN_K024_focal":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/accumulative-scratch/bioclip2_2025-06-29-23-57-16_common_name/full/log/log.txt",
#         #     "gamma_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/accumulative-scratch/bioclip2_2025-07-07-20-25-34_common_name/full/log/log.txt",
#         #     "gamma_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/accumulative-scratch/bioclip2_2025-07-07-20-28-02_common_name/full/log/log.txt",
#         #     "gamma_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/accumulative-scratch/bioclip2_2025-07-07-20-29-17_common_name/full/log/log.txt",
#         #     "gamma_1.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/accumulative-scratch/bioclip2_2025-07-07-20-32-32_common_name/full/log/log.txt",
#         #     "gamma_2.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/accumulative-scratch/bioclip2_2025-07-07-20-35-01_common_name/full/log/log.txt",
#         #     "gamma_5.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/accumulative-scratch/bioclip2_2025-07-07-20-36-01_common_name/full/log/log.txt"
#         # }

#         # "APN_K024_common_focal":{
#         #     "CE" : "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-06-30-00-01-23_common_name/full/log/log.txt",
#         #     "gamma_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/percentage-1/bioclip2_2025-07-07-20-37-34_common_name/full/log/log.txt",
#         #     "gamma_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/percentage-1/bioclip2_2025-07-07-20-38-29_common_name/full/log/log.txt",
#         #     "gamma_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/percentage-1/bioclip2_2025-07-07-20-40-01_common_name/full/log/log.txt",
#         #     "gamma_1.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/percentage-1/bioclip2_2025-07-07-20-41-09_common_name/full/log/log.txt",
#         #     "gamma_2.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/percentage-1/bioclip2_2025-07-07-20-42-33_common_name/full/log/log.txt",
#         #     "gamma_5.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/focal/percentage-1/bioclip2_2025-07-07-20-53-47_common_name/full/log/log.txt"
#         # }

#         # "APN_K024_common_cb_ce":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/accumulative-scratch/bioclip2_2025-06-29-23-57-16_common_name/full/log/log.txt",
#         #     "beta_0.99": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-ce/accumulative-scratch/bioclip2_2025-07-07-21-18-43_common_name/full/log/log.txt",
#         #     "dynamic_beta": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-ce/accumulative-scratch/bioclip2_2025-07-07-21-20-44_common_name/full/log/log.txt"
#         # }

#         #         "APN_K024_common_cb_ce":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-06-30-00-01-23_common_name/full/log/log.txt",
#         #     "beta_0.99": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-ce/percentage-1/bioclip2_2025-07-07-21-21-44_common_name/full/log/log.txt",
#         #     "dynamic_beta": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-ce/percentage-1/bioclip2_2025-07-07-21-23-04_common_name/full/log/log.txt"
#         # }

#         # "APN_K024_common_cb_focal":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/accumulative-scratch/bioclip2_2025-06-29-23-57-16_common_name/full/log/log.txt",
#         #     "dynamic_beta": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/accumulative-scratch/bioclip2_2025-07-07-20-56-34_common_name/full/log/log.txt",
#         #     "beta_0.9": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/accumulative-scratch/bioclip2_2025-07-07-20-58-12_common_name/full/log/log.txt",
#         #     "beta_0.99": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/accumulative-scratch/bioclip2_2025-07-07-20-59-36_common_name/full/log/log.txt",
#         #     "beta_0.999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/accumulative-scratch/bioclip2_2025-07-07-21-01-35_common_name/full/log/log.txt",
#         #     "beta_0.9999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/accumulative-scratch/bioclip2_2025-07-07-21-02-33_common_name/full/log/log.txt"
#         # }

#         #         "APN_K024_common_cb_focal":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-06-30-00-01-23_common_name/full/log/log.txt",
#         #     "dynamic_beta": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/percentage-1/bioclip2_2025-07-07-21-04-07_common_name/full/log/log.txt",
#         #     "beta_0.9": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/percentage-1/bioclip2_2025-07-07-21-06-05_common_name/full/log/log.txt",
#         #     "beta_0.99": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/percentage-1/bioclip2_2025-07-07-21-07-09_common_name/full/log/log.txt",
#         #     "beta_0.999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/percentage-1/bioclip2_2025-07-07-21-07-45_common_name/full/log/log.txt",
#         #     "beta_0.9999": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cb-focal/percentage-1/bioclip2_2025-07-07-21-08-47_common_name/full/log/log.txt"
#         # }

#         # "APN_K024_common_ldam":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/accumulative-scratch/bioclip2_2025-06-29-23-57-16_common_name/full/log/log.txt",
#         #     "LDAM": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ldam/accumulative-scratch/bioclip2_2025-07-07-21-30-29_common_name/full/log/log.txt"
#         # }

#         #         "APN_K024_common_bsm":{
#         #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-06-30-00-01-23_common_name/full/log/log.txt",
#         #     "BSM": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/bsm/percentage-1/bioclip2_2025-07-07-21-25-18_common_name/full/log/log.txt"
#         # }
#         "test": {
#             "1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/idaho_idaho_16/upper_bound/2025-07-10-06-43-52/bioclip2_2025-07-10-06-44-01_common_name/full/2025-07-10-06-44-01/log/log.txt"
#         }


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

#     for i, acc in enumerate(all_balanced_acc):
#         lb = all_name[i]
#         lb = lb.replace("30:", "")
#         lb = lb.replace("cb_log:", "")
#         lb = lb.replace("b0.9", "balanced-ce")
        
#         # # Set colors based on bioclip type
#         # color = None
#         # if 'zs' in all_name[i].lower():
#         #     color = 'orange'
#         # elif 'accu' in all_name[i].lower():
#         #     color = 'blue'
        
#         # Set colors for each legend
#         colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
#         color = colors[i % len(colors)]  # Cycle through colors if more than 10 lines
        
#         # Set line style: dashed for acc(cdt...) experiments, solid for others
#         linestyle = '--' if 'cdt' in all_name[i] else '-'
        
#         plt.plot(acc, label=lb, color=color, linestyle=linestyle)
#     plt.xlabel("Ckp")
#     plt.ylabel("Balanced Acc")
#     plt.title(f"{dset}(upper)_Balanced_Acc")
#     plt.legend(
#         loc='center left',
#         bbox_to_anchor=(1, 0.5)
#     )
#     plt.grid()
#     plt.savefig(f"figure_imb/{dset}(upper)_Balanced_Acc_.png", bbox_inches='tight')
#     plt.close()

# # for dset, file_list in accu_eval_path.items():
# #     print(f"Dataset: {dset}")
# #     all_balanced_acc = []
# #     all_name = []
# #     zs_balanced_acc = None
# #     zs_name = None
    
# #     for exp_name, file_path in file_list.items():
# #         if "regular" in exp_name:
# #             zs_acc, zs_balanced_acc = read_accu(file_path)
# #             zs_name = exp_name
# #         else:
# #             acc, balanced_acc = read_accu_eval(file_path)
# #             all_balanced_acc.append(balanced_acc)
# #             all_name.append(exp_name)

#     # Plot Balanced Acc
#     plt.figure(figsize=(5, 3))
    
#     # Define colors for each curve
#     colors = ['blue', 'lightblue', 'red', 'green', 'purple', 'brown', 'pink', 'orange', 'olive', 'cyan']
    
#     # # Plot ZS baseline if it exists
#     # if zs_balanced_acc is not None:
#     #     zs_lb = zs_name.replace("30:", "").replace("cb_log:", "").replace("b0.9", "balanced-ce")
#     #     plt.plot(zs_balanced_acc, label=f"{zs_lb}", color='gray', linestyle='--')
    
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

#     plt.xlabel("Ckp")
#     plt.ylabel("Balanced Acc")
#     plt.title(f"{dset}")
#     plt.legend(
#         loc='center left',
#         bbox_to_anchor=(1, 0.5)
#     )
#     plt.grid()
#     plt.savefig(f"figure_test/{dset}balanced_acc.png", bbox_inches='tight')
#     plt.close()

import os
from matplotlib import pyplot as plt

# os.chdir("/home/zhang.14217/bioclip-dev")
os.chdir("/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/")

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

file_paths = {
    # "nz_EFH_HCAME04_test-all": {
    #     "ce": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-09-23-36-41/bioclip2_2025-07-09-23-36-49_scientific_name/full/2025-07-09-23-36-49/log/log.txt",
    #     # "cdt_0.05": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-10-00-42-54/bioclip2_2025-07-10-00-43-06_common_name/full/2025-07-10-00-43-06/log/log.txt",
    #     "cdt_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-10-00-49-58/bioclip2_2025-07-10-00-50-10_common_name/full/2025-07-10-00-50-10/log/log.txt",
    #     "cdt_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-10-00-52-02/bioclip2_2025-07-10-00-52-12_common_name/full/2025-07-10-00-52-12/log/log.txt",
    #     "cdt_0.3": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-10-00-52-37/bioclip2_2025-07-10-00-52-47_common_name/full/2025-07-10-00-52-47/log/log.txt",
    #     # "cdt_0.4": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-10-00-44-55/bioclip2_2025-07-10-00-45-08_common_name/full/2025-07-10-00-45-08/log/log.txt",
    #     "cdt_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-10-00-53-02/bioclip2_2025-07-10-00-53-11_common_name/full/2025-07-10-00-53-11/log/log.txt",
    # },

    "APN_K024_test_all": {
        "ce": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-07-09-18-17-55_common_name/full/log/log.txt",
        "cdt_0.05": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-09-22-03-36_common_name/full/log/log.txt",        
        "cdt_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-09-22-06-33_common_name/full/log/log.txt",
        "cdt_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-09-22-07-38_common_name/full/log/log.txt",
        "cdt_0.3": "//users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-09-22-08-31_common_name/full/log/log.txt",
        "cdt_0.4": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-09-22-10-01_common_name/full/log/log.txt",
        "cdt_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/cdt/percentage-1/bioclip2_2025-07-09-22-11-05_common_name/full/log/log.txt",
    },
    # "ENO_B06": {
    #     "Accmu": "log/pipeline/ENO_B06/ce/accumulative-scratch/log/log.txt",
    #     "Full-Training": "log/pipeline/ENO_B06/ce/percentage-1/log/log.txt",
    #     "ZS": "log/pipeline/ENO_B06/ce/zs/log/log.txt",
    #     "Regular": "log/pipeline/ENO_B06/ce/regular/log/log.txt",
    #     "Replay": "log/pipeline/ENO_B06/ce/Replay/log/log.txt",
    #     "LWF": "log/pipeline/ENO_B06/ce/LWF/log/log.txt",
    # },
    # "PLN_D01": {
    #     "Accmu": "log/pipeline/PLN_D01/ce/accumulative-scratch/log/log.txt",
    #     "Full-Training": "log/pipeline/PLN_D01/ce/percentage-1/log/log.txt",
    #     "ZS": "log/pipeline/PLN_D01/ce/zs/log/log.txt",
    #     "Regular": "log/pipeline/PLN_D01/ce/regular/log/log.txt",
    # }
    # If you have more datasets, add them here following the same structure
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
    
    # Define colors for different lines
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, acc in enumerate(all_balanced_acc):
        lb = all_name[i]
        lb = lb.replace("30:", "")
        lb = lb.replace("cb_log:", "")
        lb = lb.replace("b0.9", "balanced-ce")
        
        # Choose color from the predefined list
        color = colors[i % len(colors)]
        
        # Use dashed line for experiments containing "cdt", solid line for others
        linestyle = '--' if 'cdt' in all_name[i].lower() else '-'
        
        plt.plot(acc, label=lb, color=color, linestyle=linestyle, linewidth=2)
        
    plt.xlabel("Ckp")
    plt.ylabel("Balanced Acc")
    plt.title(f"{dset} Balanced Acc")
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    plt.grid()
    plt.savefig(f"figure_test/{dset}_cdt_test_all_balanced_acc.png", bbox_inches='tight')
    plt.close()