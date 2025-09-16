#!/usr/bin/env python3
"""
Script to create CSV tables for different AL methods with balanced accuracy results.
"""

import json
import os
import pandas as pd
from pathlib import Path
import glob

def load_dataset_list(ml_list_path):
    """Load dataset names from ML_list.txt"""
    with open(ml_list_path, 'r') as f:
        datasets = [line.strip() for line in f if line.strip()]
    return datasets

def find_json_files(base_path, dataset, method_pattern):
    """Find JSON files matching the pattern for a dataset and method"""
    dataset_path = os.path.join(base_path, dataset)
    pattern = os.path.join(dataset_path, method_pattern, "log/final_training_summary.json")
    return glob.glob(pattern)

def extract_balanced_accuracy(json_path):
    """Extract average balanced accuracy from JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('averages', {}).get('balanced_accuracy', None)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def create_method_csv(datasets, base_path, method_config, output_file):
    """Create CSV for a specific method with different percentages"""
    
    method_name = method_config['name']
    pattern_template = method_config['pattern']
    percentages = method_config['percentages']
    
    print(f"Creating CSV for method: {method_name}")
    
    # Initialize results dictionary
    results = {'Dataset': []}
    
    # Add percentage columns
    for pct in percentages:
        results[f'{pct}%'] = []
    
    # Add ZS and UB columns
    results['ZS'] = []
    results['UB'] = []
    
    # Process each dataset
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        dataset = dataset.replace('/', '_', 1)
        results['Dataset'].append(dataset.replace('/', '_'))
        
        # Process each percentage
        for pct in percentages:
            pattern = pattern_template.format(percentage=pct)
            json_files = find_json_files(base_path, dataset, pattern)
            
            if json_files:
                # Take the first matching file
                bal_acc = extract_balanced_accuracy(json_files[0])
                results[f'{pct}%'].append(bal_acc if bal_acc is not None else 'N/A')
                print(f"  {pct}%: {bal_acc}")
            else:
                results[f'{pct}%'].append('N/A')
                print(f"  {pct}%: N/A (file not found)")
        
        # Process Zero-shot (ZS)
        zs_pattern = "zs_ce_loss/bioclip2/full_text_head"
        zs_files = find_json_files(base_path, dataset, zs_pattern)
        if zs_files:
            zs_bal_acc = extract_balanced_accuracy(zs_files[0])
            results['ZS'].append(zs_bal_acc if zs_bal_acc is not None else 'N/A')
            print(f"  ZS: {zs_bal_acc}")
        else:
            results['ZS'].append('N/A')
            print(f"  ZS: N/A (file not found)")
        
        # Process Upper Bound (UB) - Full fine-tuning
        ub_pattern = "accum_lora_bsm_loss/bioclip2/lora_8_text_head/all"
        ub_files = find_json_files(base_path, dataset, ub_pattern)
        if not ub_files:
            # Try alternative UB pattern
            ub_pattern = "upper_bound_lora_ce_loss/bioclip2/lora_8_text_head"
            ub_files = find_json_files(base_path, dataset, ub_pattern)
        
        if ub_files:
            ub_bal_acc = extract_balanced_accuracy(ub_files[0])
            results['UB'].append(ub_bal_acc if ub_bal_acc is not None else 'N/A')
            print(f"  UB: {ub_bal_acc}")
        else:
            results['UB'].append('N/A')
            print(f"  UB: N/A (file not found)")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved {method_name} results to {output_file}")
    return df

def main():
    # Configuration
    BASE_PATH = "/fs/scratch/PAS2099/camera-trap-final/AL_logs"
    ML_LIST_PATH = "/users/PAS2099/mino/ICICLE/ML_list.txt"
    OUTPUT_DIR = "/users/PAS2099/mino/ICICLE/csv"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load datasets
    datasets = load_dataset_list(ML_LIST_PATH)
    print(f"Loaded {len(datasets)} datasets")
    
    # Define method configurations
    methods = [
        {
            'name': 'KMS',
            'pattern': 'accum_lora_bsm_loss/bioclip2/lora_8_text_head/kms_percentage_{percentage}_num_samples_per_cluster_1',
            'percentages': ['0.1', '0.15', '0.2', '0.25']
        },
        {
            'name': 'ALOE',
            'pattern': 'accum_lora_bsm_loss/bioclip2/lora_8_text_head/aloe_percentage_{percentage}_num_samples_per_cluster_1',
            'percentages': ['0.1', '0.15', '0.2', '0.25']
        },
        {
            'name': 'Random',
            'pattern': 'accum_lora_bsm_loss/bioclip2/lora_8_text_head/cls_random_percentage_{percentage}',
            'percentages': ['0.1', '0.15', '0.2', '0.25']
        },
        {
            'name': 'MSP',
            'pattern': 'accum_lora_bsm_loss/bioclip2/lora_8_text_head/msp_percentage_{percentage}',
            'percentages': ['0.1', '0.15', '0.2', '0.25']
        }
    ]
    
    # Create CSV for each method
    all_results = {}
    for method_config in methods:
        output_file = os.path.join(OUTPUT_DIR, f"{method_config['name'].lower()}_results.csv")
        df = create_method_csv(datasets, BASE_PATH, method_config, output_file)
        all_results[method_config['name']] = df
    
    # Create a summary CSV with all methods (using 10% as example)
    print("\nCreating summary CSV with 10% results for all methods...")
    summary_results = {'Dataset': [dataset.replace('/', '_') for dataset in datasets]}
    
    for method_name, df in all_results.items():
        if '10%' in df.columns:
            summary_results[f'{method_name}_10%'] = df['10%'].tolist()
        else:
            summary_results[f'{method_name}_10%'] = ['N/A'] * len(datasets)
    
    # Add ZS and UB from the first method's results
    if all_results:
        first_method = list(all_results.values())[0]
        summary_results['ZS'] = first_method['ZS'].tolist()
        summary_results['UB'] = first_method['UB'].tolist()
    
    summary_df = pd.DataFrame(summary_results)
    summary_file = os.path.join(OUTPUT_DIR, "summary_10_percent.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary results to {summary_file}")
    
    print(f"\nAll CSV files saved to: {OUTPUT_DIR}")
    print("Files created:")
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('.csv'):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
