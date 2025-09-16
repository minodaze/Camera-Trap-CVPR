import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

percentages = [0.1, 0.15, 0.2, 0.25]
methods = {
    'cls_random': 'cls_random_percentage_{p}', 
    'kms': 'kms_percentage_{p}_num_samples_per_cluster_1', 
    'msp': 'msp_percentage_{p}', 
    'aloe': 'aloe_percentage_{p}_num_samples_per_cluster_1'
}
base_path = "/fs/scratch/PAS2099/camera-trap-final/AL_logs"
json_path = '{base_path}/{dataset_name}/accum_lora_bsm_loss/bioclip2/lora_8_text_head/{method}/log/final_training_summary.json'
zs_json_path = '{base_path}/{dataset_name}/zs_ce_loss/bioclip2/full_text_head/log/final_training_summary.json'
ub_json_path = '{base_path}/{dataset_name}/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log/final_training_summary.json'
# /fs/scratch/PAS2099/camera-trap-final/AL_logs/KGA_KGA_KHOGB07/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log/final_training_summary.json
def load_method_results(base_path, dataset_name):
    results = {}

    # Load results for each method
    for method in methods:
        for p in percentages:
            method = method.format(p=p)
            
        # AL_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/kms_percentage_0.9_num_samples_per_cluster_1/log/final_training_summary.json
        # AL_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/msp_percentage_0.1/log/final_training_summary.json
        # AL_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/aloe_percentage_0.6_num_samples_per_cluster_1/log/final_training_summary.json
        json_path = os.path.join(base_path, f"{dataset_name}/accum_lora_bsm_loss/bioclip2/lora_8_text_head/{method}/log/final_training_summary.json")
        print(f"Checking method {method} at path: {json_path}")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results[method] = json.load(f)
            print(f"Successfully loaded {method}")
        else:
            print(f"Warning: File not found: {json_path}")
    
    # Load zero-shot results
    zs_path = os.path.join(base_path, f"{dataset_name}/zs_ce_loss/bioclip2/full_text_head/log/final_training_summary.json")
    ub_path = os.path.join(base_path, f"{dataset_name}/accum_lora_bsm_loss/bioclip2/lora_8_text_head/all/log/final_training_summary.json")
    if os.path.exists(zs_path):
        with open(zs_path, 'r') as f:
            results['zero_shot'] = json.load(f)
    else:
        print(f"Warning: Zero-shot file not found: {zs_path}")
    
    if os.path.exists(ub_path):
        with open(ub_path, 'r') as f:
            results['upper_bound'] = json.load(f)
    else:
        print(f"Warning: Upper-bound file not found: {ub_path}")
    
    return results

def plot_methods_balanced_accuracy_curves(results, dataset_name, save_path=None):
    """Plot balanced accuracy curves for different training settings."""
    plt.figure(figsize=(12, 8))
    
    print(f"Plotting {dataset_name} with methods: {list(results.keys())}")
    
    # Define colors and styles - ensuring Upper-bound and fraction 0.1 have different colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#bcbd22']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Extract checkpoint numbers (assuming they're in format "ckp_X")
    if results:
        first_key = list(results.keys())[0]
        checkpoints = list(results[first_key]['checkpoint_results'].keys())
        ckp_numbers = [int(ckp.split('_')[1]) for ckp in checkpoints]
    
    method_plot_idx = 0
    
    # Plot each training setting with specific color assignments
    for setting, data in results.items():
        print(f"Processing setting: {setting}")
        if 'checkpoint_results' in data:
            balanced_accs = []
            
            for ckp in checkpoints:
                if ckp in data['checkpoint_results']:
                    balanced_accs.append(data['checkpoint_results'][ckp]['balanced_accuracy'])
                else:
                    balanced_accs.append(np.nan)
            
            # Create label and assign specific colors
            if setting == 'zero_shot':
                label = 'Zero-Shot'
                color = colors[0]  # Blue
                linestyle = linestyles[0]
            elif setting == 'upper_bound':
                label = 'ALL Data (Upper-Bound)'
                color = colors[1]  # Orange
                linestyle = linestyles[1]
            else:
                method = setting.split('_')[0].upper()  # Extract method name and capitalize
                label = f'{method}' if method != 'CLS' else 'Random class'
                if method == 'MSP':
                    label = f'{percentage*100}% confidence score'
                elif method == 'KMS':
                    label = f'{percentage*100}% KMeans'
                elif method == 'FR':
                    label = f'FR'
                elif method == 'ALOE':
                    label = f'{percentage*100}% ALOE'
                elif method == 'CLS':
                    label = f'{percentage*100}% class balanced random selection'
                else:
                    label = f'{method}'

                # Start from index 2 for AL methods to avoid conflict with zero-shot and upper-bound
                color = colors[2 + method_plot_idx]
                linestyle = linestyles[2 + method_plot_idx]
                method_plot_idx += 1
            
            print(f"Plotting {setting} with label '{label}' and color {color}")
            print(f"  Data points: {balanced_accs}")
            plt.plot(ckp_numbers, balanced_accs, 
                    color=color,
                    linestyle=linestyle,
                    marker='o', markersize=6, linewidth=2,
                    label=label)
        else:
            print(f"Skipping {setting} - no checkpoint_results found")
    
    plt.xlabel('Checkpoint', fontsize=14)
    plt.ylabel('Balanced Accuracy', fontsize=14)
    plt.title(f'{dataset_name}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Set y-axis limits for better visualization
    plt.ylim(0, 1.1)
    
    # Customize x-axis
    plt.xticks(ckp_numbers)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def load_json_results(base_path, dataset_name):
    """Load all JSON results for a given dataset."""
    results = {}
    
    # Define the fractions to look for
    fractions = ['0.1', '0.3', '0.5', '0.7', '0.9']
    
    # Load results for each fraction
    for frac in fractions:
        # AL_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/kms_percentage_0.9_num_samples_per_cluster_1/log/final_training_summary.json
        # AL_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/msp_percentage_0.1/log/final_training_summary.json
        # AL_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/aloe_percentage_0.6_num_samples_per_cluster_1/log/final_training_summary.json
        json_path = os.path.join(base_path, f"{dataset_name}/accum_lora_bsm_loss/bioclip2/lora_8_text_head/log/final_training_summary.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results[f'frac_{frac}'] = json.load(f)
        else:
            print(f"Warning: File not found: {json_path}")
    
    # Load zero-shot results
    zs_path = os.path.join(base_path, f"{dataset_name}/zs_ce_loss/bioclip2/full_text_head/log/final_training_summary.json")
    # /fs/scratch/PAS2099/camera-trap-final/AL_logs/orinoquia_orinoquia_N29/accum_lora_bsm_loss/bioclip2/lora_8_text_head/log/final_training_summary.json
    ub_path = os.path.join(base_path, f"{dataset_name}/accum_lora_bsm_loss/bioclip2/lora_8_text_head/log/final_training_summary.json")
    if os.path.exists(zs_path):
        with open(zs_path, 'r') as f:
            results['zero_shot'] = json.load(f)
    else:
        print(f"Warning: Zero-shot file not found: {zs_path}")
    
    if os.path.exists(ub_path):
        with open(ub_path, 'r') as f:
            results['upper_bound'] = json.load(f)
    else:
        print(f"Warning: Upper-bound file not found: {ub_path}")
    
    return results

def plot_balanced_accuracy_curves(results, dataset_name, save_path=None):
    """Plot balanced accuracy curves for different training settings."""
    plt.figure(figsize=(12, 8))
    
    # Define colors and styles - ensuring Upper-bound and fraction 0.1 have different colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Extract checkpoint numbers (assuming they're in format "ckp_X")
    if results:
        first_key = list(results.keys())[0]
        checkpoints = list(results[first_key]['checkpoint_results'].keys())
        ckp_numbers = [int(ckp.split('_')[1]) for ckp in checkpoints]
    
    plot_idx = 0
    
    # Plot each training setting with specific color assignments
    for setting, data in results.items():
        if 'checkpoint_results' in data:
            balanced_accs = []
            
            for ckp in checkpoints:
                if ckp in data['checkpoint_results']:
                    balanced_accs.append(data['checkpoint_results'][ckp]['balanced_accuracy'])
                else:
                    balanced_accs.append(np.nan)
            
            # Create label and assign specific colors
            if setting == 'zero_shot':
                label = 'Zero-Shot'
                color = colors[0]  # Blue
            elif setting == 'upper_bound':
                label = 'ALL Data (Upper-Bound)'
                color = colors[1]  # Orange
            else:
                frac_value = setting.split('_')[1]
                label = f'Ratio {frac_value}'
                # Start from index 2 for fractions to avoid conflict with zero-shot and upper-bound
                color = colors[2 + plot_idx]
            
            plt.plot(ckp_numbers, balanced_accs, 
                    color=color,
                    linestyle=linestyles[plot_idx % len(linestyles)],
                    marker='o', markersize=6, linewidth=2,
                    label=label)
            plot_idx += 1
    
    plt.xlabel('Checkpoint', fontsize=14)
    plt.ylabel('Balanced Accuracy', fontsize=14)
    plt.title(f'{dataset_name}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Set y-axis limits for better visualization
    plt.ylim(0, 1.1)
    
    # Customize x-axis
    plt.xticks(ckp_numbers)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_average_performance_comparison(all_results, save_path=None):
    """Plot average balanced accuracy across all datasets for different OOD fractions."""
    plt.figure(figsize=(10, 6))
    
    fractions = ['0.3', '0.5']
    
    # Calculate average performance for each fraction across all datasets
    avg_balanced_accs = []
    std_balanced_accs = []
    
    for frac in fractions:
        frac_key = f'frac_{frac}'
        balanced_accs = []
        
        for dataset_name, results in all_results.items():
            if frac_key in results:
                avg_acc = results[frac_key]['averages']['balanced_accuracy']
                balanced_accs.append(avg_acc)
        
        if balanced_accs:
            avg_balanced_accs.append(np.mean(balanced_accs))
            std_balanced_accs.append(np.std(balanced_accs))
        else:
            avg_balanced_accs.append(np.nan)
            std_balanced_accs.append(0)
    
    # Add zero-shot performance
    zs_balanced_accs = []
    for dataset_name, results in all_results.items():
        if 'zero_shot' in results:
            avg_acc = results['zero_shot']['averages']['balanced_accuracy']
            zs_balanced_accs.append(avg_acc)
    
    if zs_balanced_accs:
        zs_avg = np.mean(zs_balanced_accs)
        zs_std = np.std(zs_balanced_accs)
    else:
        zs_avg, zs_std = np.nan, 0
    
    # Convert fractions to numeric for plotting
    frac_numeric = [float(f) for f in fractions]
    
    # Plot the curve
    plt.errorbar(frac_numeric, avg_balanced_accs, yerr=std_balanced_accs,
                marker='o', markersize=8, linewidth=2, capsize=5,
                label='OOD Training')
    
    # Plot zero-shot as a horizontal line
    if not np.isnan(zs_avg):
        plt.axhline(y=zs_avg, color='red', linestyle='--', linewidth=2, 
                   label=f'Zero-Shot (avg: {zs_avg:.3f})')
        plt.fill_between(frac_numeric, zs_avg - zs_std, zs_avg + zs_std, 
                        color='red', alpha=0.2)
    
    plt.xlabel('OOD Fraction', fontsize=14)
    plt.ylabel('Average Balanced Accuracy', fontsize=14)
    plt.title('Average Balanced Accuracy vs Different percentage\n(Across All Datasets)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average performance plot saved to: {save_path}")
    
    plt.show()

def main():
    # Configuration
    base_path = "/fs/scratch/PAS2099/camera-trap-final/AL_logs"
    # json_path = "/fs/scratch/PAS2099/camera-trap-final/randomOOD_logs/KGA_KGA_KHOLA08/upper_bound_ce_loss_frac0.7/bioclip2/full_text_head/cls_random_random_fraction_0.7/log/final_training_summary.json"
    # Read ML_list.txt to get dataset names
    ml_list_path = "ML_list.txt"
    if os.path.exists(ml_list_path):
        with open(ml_list_path, 'r') as f:
            datasets = [line.replace("/", "_", 1).strip() for line in f.readlines() if line.strip()]
    else:
        # If ML_list.txt doesn't exist, try to find datasets automatically
        print("ML_list.txt not found, searching for datasets automatically...")
        datasets = []
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, item)):
                    datasets.append(item)
    
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    # Load all results
    all_results = {}
    for dataset in datasets:
        print(f"Loading results for {dataset}...")
        results = load_method_results(base_path, dataset)
        if results:
            all_results[dataset] = results
        else:
            print(f"No results found for {dataset}")
    
    # Create output directory
    output_dir = f"{percentage}_Diffmethod_accum_AL_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot individual dataset curves
    for dataset_name, results in all_results.items():
        print(f"Plotting curves for {dataset_name}...")
        save_path = os.path.join(output_dir, f"{dataset_name}_balanced_accuracy_curves.png")
        plot_methods_balanced_accuracy_curves(results, dataset_name, save_path)
    
    # Plot average performance comparison
    # if all_results:
    #     print("Plotting average performance comparison...")
    #     save_path = os.path.join(output_dir, "average_balanced_accuracy_comparison.png")
    #     plot_average_performance_comparison(all_results, save_path)
    
    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()

