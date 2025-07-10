

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
        results.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
        
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

    "APN_K024":{
        "upper_bound(petl_load)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-06-30-00-01-23_common_name/full/log/log.txt",
        "zs(openclip_load)": "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/APN_K024/ce/percentage-1/bioclip2_2025-07-09-18-17-55_common_name/full/log/log.txt"
    }
}


if __name__ == "__main__":
    # Change to the correct directory
    os.chdir("/users/PAS2119/hou/ICICLE/ICICLE-Benchmark")
    
    # Calculate and print mean balanced accuracies
    calculate_mean_balanced_acc(file_paths)

# %%
