#!/usr/bin/env python3
"""
Script to create CSV tables for different AL methods with balanced accuracy results.
No external dependencies - uses only standard library.
"""

import json
import os
import csv
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

def format_accuracy(acc):
    """Format accuracy value for CSV output"""
    if acc is None:
        return 'N/A'
    return f"{acc:.4f}"

def create_method_csv(datasets, base_path, method_config, output_file):
    """Create CSV for a specific method with different percentages"""
    
    method_name = method_config['name']
    pattern_template = method_config['pattern']
    percentages = method_config['percentages']
    
    print(f"Creating CSV for method: {method_name}")
    
    # Prepare CSV data
    csv_data = []
    
    # Header row
    header = ['Dataset'] + [f'{pct}%' for pct in percentages] + ['ZS', 'UB']
    csv_data.append(header)
    
    # Process each dataset
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        row = [dataset.replace('/', '_')]
        
        # Process each percentage
        for pct in percentages:
            pattern = pattern_template.format(percentage=pct)
            json_files = find_json_files(base_path, dataset, pattern)
            
            if json_files:
                # Take the first matching file
                bal_acc = extract_balanced_accuracy(json_files[0])
                row.append(format_accuracy(bal_acc))
                print(f"  {pct}%: {format_accuracy(bal_acc)}")
            else:
                row.append('N/A')
                print(f"  {pct}%: N/A (file not found)")
        
        # Process Zero-shot (ZS)
        zs_pattern = "zs/bioclip2/full_text_head"
        zs_files = find_json_files(base_path, dataset, zs_pattern)
        if zs_files:
            zs_bal_acc = extract_balanced_accuracy(zs_files[0])
            row.append(format_accuracy(zs_bal_acc))
            print(f"  ZS: {format_accuracy(zs_bal_acc)}")
        else:
            row.append('N/A')
            print(f"  ZS: N/A (file not found)")
        
        # Process Upper Bound (UB) - Try multiple patterns
        ub_patterns = [
            "upper_bound_ce_loss/bioclip2/full_text_head",
            "upper_bound_lora_ce_loss/bioclip2/lora_8_text_head",
            "upper_bound_bsm_loss/log"  # Alternative pattern
        ]
        
        ub_bal_acc = None
        for ub_pattern in ub_patterns:
            ub_files = find_json_files(base_path, dataset, ub_pattern)
            if ub_files:
                ub_bal_acc = extract_balanced_accuracy(ub_files[0])
                break
        
        row.append(format_accuracy(ub_bal_acc))
        print(f"  UB: {format_accuracy(ub_bal_acc)}")
        
        csv_data.append(row)
    
    # Write CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
    
    print(f"Saved {method_name} results to {output_file}")
    return csv_data

def create_summary_csv(all_csv_data, output_file, summary_percentage='0.1'):
    """Create a summary CSV with specified percentage for all methods"""
    print(f"\nCreating summary CSV with {summary_percentage} results for all methods...")
    
    # Find datasets from the first method
    datasets = [row[0] for row in all_csv_data[list(all_csv_data.keys())[0]][1:]]  # Skip header
    
    # Prepare summary data
    summary_data = []
    header = ['Dataset']
    
    # Add method columns
    for method_name in all_csv_data.keys():
        header.append(f'{method_name}_{summary_percentage}')
    
    # Add ZS and UB columns
    header.extend(['ZS', 'UB'])
    summary_data.append(header)
    
    # Process each dataset
    for i, dataset in enumerate(datasets):
        row = [dataset]
        
        # Add results for each method
        for method_name, csv_data in all_csv_data.items():
            method_header = csv_data[0]  # Header row
            method_row = csv_data[i + 1]  # Data row (i+1 because header is at index 0)
            
            # Find the column for the specified percentage
            pct_col = f'{float(summary_percentage)*100:.0f}%'
            if pct_col in method_header:
                col_idx = method_header.index(pct_col)
                row.append(method_row[col_idx])
            else:
                row.append('N/A')
        
        # Add ZS and UB from the first method
        first_method_data = list(all_csv_data.values())[0]
        first_method_header = first_method_data[0]
        first_method_row = first_method_data[i + 1]
        
        # ZS column
        if 'ZS' in first_method_header:
            zs_idx = first_method_header.index('ZS')
            row.append(first_method_row[zs_idx])
        else:
            row.append('N/A')
        
        # UB column
        if 'UB' in first_method_header:
            ub_idx = first_method_header.index('UB')
            row.append(first_method_row[ub_idx])
        else:
            row.append('N/A')
        
        summary_data.append(row)
    
    # Write summary CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(summary_data)
    
    print(f"Saved summary results to {output_file}")

def main():
    # Configuration
    BASE_PATH = "/fs/scratch/PAS2099/camera-trap-final/AL_logs"
    ML_LIST_PATH = "/users/PAS2099/mino/ICICLE/ML_list.txt"
    OUTPUT_DIR = "/users/PAS2099/mino/ICICLE/results_csv"
    
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
            'percentages': ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3']
        },
        {
            'name': 'ALOE',
            'pattern': 'accum_lora_bsm_loss/bioclip2/lora_8_text_head/aloe_percentage_{percentage}_num_samples_per_cluster_1',
            'percentages': ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3']
        },
        {
            'name': 'Random',
            'pattern': 'accum_lora_bsm_loss/bioclip2/lora_8_text_head/random_percentage_{percentage}',
            'percentages': ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3']
        },
        {
            'name': 'ActiveFT',
            'pattern': 'accum_lora_bsm_loss/bioclip2/lora_8_text_head/activeft_percentage_{percentage}',
            'percentages': ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3']
        }
    ]
    
    # Create CSV for each method
    all_csv_data = {}
    for method_config in methods:
        output_file = os.path.join(OUTPUT_DIR, f"{method_config['name'].lower()}_results.csv")
        csv_data = create_method_csv(datasets, BASE_PATH, method_config, output_file)
        all_csv_data[method_config['name']] = csv_data
    
    # Create summary CSVs for different percentages
    summary_percentages = ['0.1', '0.2', '0.3']
    for pct in summary_percentages:
        summary_file = os.path.join(OUTPUT_DIR, f"summary_{pct.replace('.', '')}_percent.csv")
        create_summary_csv(all_csv_data, summary_file, pct)
    
    print(f"\nAll CSV files saved to: {OUTPUT_DIR}")
    print("Files created:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        if file.endswith('.csv'):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
