#!/usr/bin/env python3
"""
Script to generate model paths from dataset names.
Reads datasets from eval_dataset.txt and generates corresponding model paths
based on the pattern observed in model_path.txt.
"""

import os

def dataset_to_model_path(dataset):
    """
    Convert dataset name to model path following the pattern:
    /fs/scratch/PAS2099/camera-trap-final/logs/{dataset_underscore}/upper_bound_bsm_loss/lr_0.000025/bioclip2/lora_8_text_head/log/pretrain_best_model.pth
    
    Args:
        dataset (str): Dataset name like "orinoquia/orinoquia_N25"
    
    Returns:
        str: Full model path
    """
    # Convert slashes to underscores for the path
    name = dataset.split("/")[-1]  # Get the last part of the dataset name

    # /fs/scratch/PAS2099/camera-trap-final/logs/na_na_archbold_FL-32/upper_bound_bsm_loss/lr_0.000025/bioclip2/full_text_head/log/pretrain_best_model.pth

    # Build the model path
    model_path = f"/fs/scratch/PAS2099/camera-trap-final/logs/{name}/{name}_accu_ce/bioclip2/full_text_head/log/"
    return model_path

def main():
    """
    Main function to read datasets and generate model paths.
    """
    # Input and output files
    input_file = "eval_dataset.txt"
    output_file = "model_path.txt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # Read datasets from eval_dataset.txt
    datasets = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                datasets.append(line)
    
    print(f"Found {len(datasets)} datasets in {input_file}")
    
    # Generate model paths
    model_paths = []
    for dataset in datasets:
        model_path = dataset_to_model_path(dataset)
        model_paths.append(model_path)
        print(f"  {dataset} -> {os.path.basename(os.path.dirname(model_path))}/pretrain_best_model.pth")
    
    # Write model paths to output file
    with open(output_file, 'w') as f:
        for model_path in model_paths:
            f.write(model_path + '\n')
    
    print(f"\nGenerated {len(model_paths)} model paths")
    print(f"Model paths written to {output_file}")
    
    # Verify a few paths exist (optional check)
    existing_count = 0
    missing_paths = []
    
    print(f"\nChecking if model files exist...")
    for i, model_path in enumerate(model_paths):  # Check first 5 only
        if os.path.exists(model_path):
            existing_count += 1
            print(f"  ✅ {datasets[i]}: Model exists")
        else:
            missing_paths.append((datasets[i], model_path))
            print(f"  ❌ {datasets[i]}: Model NOT found")
    
    if missing_paths:
        print(f"\nWarning: {len(missing_paths)} model files not found (from first 5 checked)")
        print("This might be expected if training is not complete.")
    else:
        print(f"\nAll checked model files exist! ({existing_count}/{len(model_paths)} checked)")

if __name__ == "__main__":
    main()
