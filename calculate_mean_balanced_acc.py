

#%%
import os
import numpy as np

def read_accu(file_path):
    """Read accuracy from log file"""
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
        acc = float(acc)
        balanced_acc = float(balanced_acc)
        acc_arr.append(acc)
        balanced_acc_arr.append(balanced_acc)
    return acc_arr, balanced_acc_arr

def calculate_mean_balanced_acc(file_paths_dict):
    """Calculate mean balanced accuracy for each experiment"""
    print("="*80)
    print("Mean Balanced Accuracy for All Experiments")
    print("="*80)
    
    for dataset_name, experiments in file_paths_dict.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 50)
        
        # Collect results for sorting
        results = []
        
        for exp_name, file_path in experiments.items():
            try:
                if os.path.exists(file_path):
                    acc, balanced_acc = read_accu(file_path)
                    if balanced_acc:
                        mean_balanced_acc = np.mean(balanced_acc)
                        results.append((exp_name, mean_balanced_acc))
                    else:
                        results.append((exp_name, None, "No data found"))
                else:
                    results.append((exp_name, None, "Log file not found"))
            except Exception as e:
                results.append((exp_name, None, f"Error reading file - {e}"))
        
        # Sort by mean balanced accuracy (descending), put None values at the end
        # results.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
        
        # Print sorted results
        for result in results:
            if len(result) == 2:  # Valid result with accuracy
                exp_name, mean_balanced_acc = result
                print(f"{exp_name}: {mean_balanced_acc:.4f}")
            else:  # Error case
                exp_name, _, error_msg = result
                print(f"{exp_name}: {error_msg}")

# Define the file paths (same as in your plot.py)
file_paths = {
    # "na_lebec_CA-24_common": {
    #     "acc(ce)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/ce/accumulative-scratch/bioclip2_2025-07-03-01-03-32_common_name/full/log/log.txt",
    #     "acc(cdt_0.05)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-19-19_common_name/full/log/log.txt",
    #     "acc(cdt_0.1)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-15-03_common_name/full/log/log.txt",
    #     "acc(cdt_0.2)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-17-00_common_name/full/log/log.txt",
    #     "acc(cdt_0.3)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-06-47_common_name/full/log/log.txt",
    #     "acc(cdt_0.4)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-17-52_common_name/full/log/log.txt",
    #     "acc(cdt_0.5)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-24/cdt/accumulative-scratch/bioclip2_2025-07-03-01-18-31_common_name/full/log/log.txt" 
    # }

    #   "ENO_C05_common_focal":{
    #     "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/accumulative-scratch/bioclip2_2025-06-30-00-06-18_common_name/full/log/log.txt",
    #     "gamma_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-24-21_common_name/full/log/log.txt",
    #     "gamma_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-26-19_common_name/full/log/log.txt",
    #     "gamma_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-27-27_common_name/full/log/log.txt",
    #     "gamma_1.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-28-44_common_name/full/log/log.txt",
    #     "gamma_2.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-32-02_common_name/full/log/log.txt",
    #     "gamma_5.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/accumulative-scratch/bioclip2_2025-07-07-17-32-55_common_name/full/log/log.txt"
    # }

#         "ENO_C05_common_focal":{
#         "CE": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
#         "gamma_0.1": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-36-29_common_name/full/log/log.txt",        
#         "gamma_0.2": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-37-59_common_name/full/log/log.txt",    
#         "gamma_0.5": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-39-31_common_name/full/log/log.txt",    
#         "gamma_1.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-40-30_common_name/full/log/log.txt",    
#         "gamma_2.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-43-34_common_name/full/log/log.txt",    
#         "gamma_5.0": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/focal/percentage-1_new_settings/bioclip2_2025-07-07-17-44-36_common_name/full/log/log.txt" 


# }

    # "ENO_C05_common": {
    #     "upper_bound(ce)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
    #     "upper_bound(cdt_0.05)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-01-10-25_common_name/full/log/log.txt",
    #     "upper_bound(cdt_0.1)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-00-51-47_common_name/full/log/log.txt",
    #     "upper_bound(cdt_0.2)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-00-54-09_common_name/full/log/log.txt",
    #     "upper_bound(cdt_0.3)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-02-23-29-06_common_name/full/log/log.txt",
    #     "upper_bound(cdt_0.4)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-00-55-34_common_name/full/log/log.txt",
    #     "upper_bound(cdt_0.5)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/cdt/percentage-1/bioclip2_2025-07-03-01-09-09_common_name/full/log/log.txt"

    # }
    #  "ENO_C05_test":{
    #         "upper_bound(petl_load)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-06-30-10-16-39_common_name/full/log/log.txt",
    #         "upper_bound(openclip_load)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_C05/ce/percentage-1/bioclip2_2025-07-09-19-07-03_common_name/full/log/log.txt"
    #     }

    "upper_bound":{
        # "nz/nz_EFH_HCAME04": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAME04/upper_bound/2025-07-09-23-36-41/bioclip2_2025-07-09-23-36-49_scientific_name/full/2025-07-09-23-36-49/log/log.txt",
        # "orinoquia/orinoquia_A06": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/orinoquia_orinoquia_A06/upper_bound/2025-07-10-05-43-18/bioclip2_2025-07-10-05-43-28_common_name/full/2025-07-10-05-43-28/log/log.txt",
        # "idaho/idaho_16": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/idaho_idaho_16/upper_bound/2025-07-10-06-43-52/bioclip2_2025-07-10-06-44-01_common_name/full/2025-07-10-06-44-01/log/log.txt",
        # "orinoquia/orinoquia_N14": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/orinoquia_orinoquia_N14/upper_bound/2025-07-10-07-58-04/bioclip2_2025-07-10-07-58-16_common_name/full/2025-07-10-07-58-16/log/log.txt",
        # "orinoquia/orinoquia_N20": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/orinoquia_orinoquia_N20/upper_bound/2025-07-09-23-40-05/bioclip2_2025-07-09-23-40-15_common_name/full/2025-07-09-23-40-15/log/log.txt",
        # "nz/nz_EFH_HCAMA02": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAMA02/upper_bound/2025-07-10-00-23-19/bioclip2_2025-07-10-00-24-11_common_name/full/2025-07-10-00-24-11/log/log.txt",
        # "nz/nz_EFH_HCAMC04": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAMC04/upper_bound/2025-07-10-03-32-05/bioclip2_2025-07-10-03-32-14_common_name/full/2025-07-10-03-32-14/log/log.txt",
        # "nz/nz_EFD_DCAMC04": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFD_DCAMC04/upper_bound/2025-07-09-23-41-34/bioclip2_2025-07-09-23-41-43_common_name/full/2025-07-09-23-41-43/log/log.txt",
        # "nz/nz_EFH_HCAMC02": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAMC02/upper_bound/2025-07-10-02-04-54/bioclip2_2025-07-10-02-05-03_common_name/full/2025-07-10-02-05-03/log/log.txt",
        # "KGA_KGA_KHOGA04": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/KGA_KGA_KHOGA04/upper_bound/2025-07-10-06-15-14/bioclip2_2025-07-10-06-15-25_common_name/full/2025-07-10-06-15-25/log/log.txt",
        # "idaho/idaho_231": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/idaho_idaho_231/upper_bound/2025-07-10-07-09-50/bioclip2_2025-07-10-07-09-58_common_name/full/2025-07-10-07-09-58/log/log.txt",
        # "orinoquia/orinoquia_N25": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/orinoquia_orinoquia_N25/upper_bound/2025-07-10-00-37-42/bioclip2_2025-07-10-00-37-54_common_name/full/2025-07-10-00-37-54/log/log.txt",
        # "nz/nz_EFH_HCAMB07": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/nz_nz_EFH_HCAMB07/upper_bound/2025-07-10-02-46-31/bioclip2_2025-07-10-02-46-41_common_name/full/2025-07-10-02-46-41/log/log.txt",
        # "orinoquia/orinoquia_A07": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/orinoquia_orinoquia_A07/upper_bound/2025-07-10-07-27-07/bioclip2_2025-07-10-07-27-17_common_name/full/2025-07-10-07-27-17/log/log.txt",
        # "KAR_KAR_B03": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/log_auto/pipeline/KAR_KAR_B03/upper_bound/2025-07-10-08-10-40/bioclip2_2025-07-10-08-10-49_common_name/full/2025-07-10-08-10-49/log/log.txt"

         "orinoquia_orinoquia_M02": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_M02/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "serengeti_serengeti_P07": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_P07/upper_bound/2025-07-12-03-37-53/bioclip2_2025-07-12-03-37-59_common_name/full/2025-07-12-03-37-59/log/log.txt",
    "nz_nz_PS1_CAM7110": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_PS1_CAM7110/upper_bound/2025-07-12-01-14-29/bioclip2_2025-07-12-01-14-39_common_name/full/2025-07-12-01-14-39/log/log.txt",
    "swg_swg_loc_0297": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0297/upper_bound/2025-07-12-07-29-49/bioclip2_2025-07-12-07-30-06_common_name/full/2025-07-12-07-30-06/log/log.txt",
    "serengeti_serengeti_D10": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_D10/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "orinoquia_orinoquia_M04": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_M04/upper_bound/2025-07-12-01-03-33/bioclip2_2025-07-12-01-03-42_common_name/full/2025-07-12-01-03-42/log/log.txt",
    "swg_swg_loc_0263": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0263/upper_bound/2025-07-12-06-32-21/bioclip2_2025-07-12-06-32-33_common_name/full/2025-07-12-06-32-33/log/log.txt",
    "serengeti_serengeti_S12": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_S12/upper_bound/2025-07-12-02-10-54/bioclip2_2025-07-12-02-11-03_common_name/full/2025-07-12-02-11-03/log/log.txt",
    "nz_nz_EFH_HCAMJ02": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_EFH_HCAMJ02/upper_bound/2025-07-12-04-31-42/bioclip2_2025-07-12-04-31-53_common_name/full/2025-07-12-04-31-53/log/log.txt",
    "serengeti_serengeti_K10": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_K10/upper_bound/2025-07-12-07-40-02/bioclip2_2025-07-12-07-40-17_common_name/full/2025-07-12-07-40-17/log/log.txt",
    "orinoquia_orinoquia_N20": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_N20/upper_bound/2025-07-12-08-25-04/bioclip2_2025-07-12-08-25-16_common_name/full/2025-07-12-08-25-16/log/log.txt",
    "nz_nz_PS1_CAM7910": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_PS1_CAM7910/upper_bound/2025-07-12-01-59-01/bioclip2_2025-07-12-01-59-13_common_name/full/2025-07-12-01-59-13/log/log.txt",
    "swg_swg_loc_0207": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0207/upper_bound/2025-07-12-05-58-45/bioclip2_2025-07-12-05-59-01_common_name/full/2025-07-12-05-59-01/log/log.txt",
    "serengeti_serengeti_N12": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_N12/upper_bound/2025-07-12-06-49-17/bioclip2_2025-07-12-06-49-34_common_name/full/2025-07-12-06-49-34/log/log.txt",
    "nz_nz_EFH_HCAMH09": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_EFH_HCAMH09/upper_bound/2025-07-12-02-54-28/bioclip2_2025-07-12-02-54-39_common_name/full/2025-07-12-02-54-39/log/log.txt",
    "nz_nz_RMP_HCAMM21": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_RMP_HCAMM21/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "swg_swg_loc_0296": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0296/upper_bound/2025-07-12-01-32-30/bioclip2_2025-07-12-01-32-39_common_name/full/2025-07-12-01-32-39/log/log.txt",
    "orinoquia_orinoquia_N13": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_N13/upper_bound/2025-07-12-05-09-51/bioclip2_2025-07-12-05-10-05_common_name/full/2025-07-12-05-10-05/log/log.txt",
    "serengeti_serengeti_I01": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_I01/upper_bound/2025-07-12-02-43-38/bioclip2_2025-07-12-02-43-48_common_name/full/2025-07-12-02-43-48/log/log.txt",
    "serengeti_serengeti_J10": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_J10/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "orinoquia_orinoquia_N14": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_N14/upper_bound/2025-07-12-05-59-50/bioclip2_2025-07-12-06-00-03_common_name/full/2025-07-12-06-00-03/log/log.txt",
    "swg_swg_loc_0605": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0605/upper_bound/2025-07-12-06-32-13/bioclip2_2025-07-12-06-32-31_common_name/full/2025-07-12-06-32-31/log/log.txt",
    "orinoquia_orinoquia_N27": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_N27/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "swg_swg_loc_0607": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0607/upper_bound/2025-07-12-09-09-50/bioclip2_2025-07-12-09-10-05_common_name/full/2025-07-12-09-10-05/log/log.txt",
    "orinoquia_orinoquia_A06": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_A06/upper_bound/2025-07-12-02-09-51/bioclip2_2025-07-12-02-09-58_common_name/full/2025-07-12-02-09-58/log/log.txt",
    "serengeti_serengeti_K11": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/serengeti_serengeti_K11/upper_bound/2025-07-12-09-03-34/bioclip2_2025-07-12-09-03-48_common_name/full/2025-07-12-09-03-48/log/log.txt",
    "seattle_seattle_location-06": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/seattle_seattle_location-06/upper_bound/2025-07-12-01-28-45/bioclip2_2025-07-12-01-28-55_common_name/full/2025-07-12-01-28-55/log/log.txt",
    "orinoquia_orinoquia_A07": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/orinoquia_orinoquia_A07/upper_bound/2025-07-12-03-10-31/bioclip2_2025-07-12-03-10-44_common_name/full/2025-07-12-03-10-44/log/log.txt",
    "nz_nz_Z20_3P3250": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_Z20_3P3250/upper_bound/2025-07-12-01-18-28/bioclip2_2025-07-12-01-18-38_common_name/full/2025-07-12-01-18-38/log/log.txt",
    "nz_nz_EFH_HCAMJ03": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_EFH_HCAMJ03/upper_bound/2025-07-12-07-56-21/bioclip2_2025-07-12-07-56-35_common_name/full/2025-07-12-07-56-35/log/log.txt",
    "swg_swg_loc_0198": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0198/upper_bound/2025-07-12-02-03-58/bioclip2_2025-07-12-02-04-08_common_name/full/2025-07-12-02-04-08/log/log.txt",
    "swg_swg_loc_0287": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0287/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "swg_swg_loc_0604": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/swg_swg_loc_0604/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "nz_nz_EFH_HCAMH08": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_EFH_HCAMH08/upper_bound/2025-07-12-00-10-47/bioclip2_2025-07-12-00-10-59_common_name/full/2025-07-12-00-10-59/log/log.txt",
    "nz_nz_LCA_EC21": "/fs/scratch/PAS2099/hou/icicle/log_auto/pipeline/nz_nz_LCA_EC21/upper_bound/2025-07-12-00-12-19/bioclip2_2025-07-12-00-12-28_common_name/full/2025-07-12-00-12-28/log/log.txt"
    }
}


if __name__ == "__main__":
    # Change to the correct directory
    os.chdir("/users/PAS2119/hou/ICICLE/ICICLE-Benchmark")
    
    # Calculate and print mean balanced accuracies
    calculate_mean_balanced_acc(file_paths)

# %%
