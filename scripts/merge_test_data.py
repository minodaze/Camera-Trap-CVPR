#!/usr/bin/env python3
"""
Script to merge all checkpoint data from test.json into a unified test_all.json
where each checkpoint contains all datapoints from the original file.
"""

import json
import os
from pathlib import Path

def merge_test_data(input_file, output_file):
    """
    Merge all checkpoint data into each checkpoint.
    
    Args:
        input_file (str): Path to the original test.json file
        output_file (str): Path to save the merged test_all.json file
    """
    # Read the original test.json file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Collect all datapoints from all checkpoints (no deduplication)
    all_datapoints = []
    
    # Iterate through all checkpoints and collect all datapoints
    for ckp_key, ckp_data in data.items():
        if isinstance(ckp_data, list):
            all_datapoints.extend(ckp_data)
    
    print(f"Found {len(all_datapoints)} total datapoints across all checkpoints")
    
    # Create new structure where each checkpoint has all datapoints
    merged_data = {}
    
    # Get all checkpoint keys from original data
    checkpoint_keys = list(data.keys())
    
    # Assign all datapoints to each checkpoint
    for ckp_key in checkpoint_keys:
        merged_data[ckp_key] = all_datapoints.copy()
        print(f"Assigned {len(all_datapoints)} datapoints to {ckp_key}")
    
    # Save the merged data
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Successfully saved merged data to {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"Original file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Number of checkpoints: {len(checkpoint_keys)}")
    print(f"Datapoints per checkpoint: {len(all_datapoints)}")
    
    return merged_data

def validate_data_format(data):
    """
    Validate that the data follows the basic expected format.
    
    Args:
        data (dict): The data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    for ckp_key, ckp_data in data.items():
        if not isinstance(ckp_data, list):
            print(f"❌ Checkpoint {ckp_key} data is not a list")
            return False
    
    print("✅ Data format validation passed")
    return True

def main():
    # Define file paths
    input_file = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/APN/APN_K024_common_name/test.json"
    output_file = "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/APN/APN_K024_common_name/test_all.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge the data
    try:
        merged_data = merge_test_data(input_file, output_file)
        
        # Validate the merged data format
        if validate_data_format(merged_data):
            print("✅ Data merging completed successfully!")
        else:
            print("❌ Data validation failed!")
            return
        
        # Optional: Print first few items from first checkpoint as verification
        first_ckp = list(merged_data.keys())[0]
        print(f"\nFirst 3 items from {first_ckp}:")
        for i, item in enumerate(merged_data[first_ckp][:3]):
            print(f"  {i+1}. Class: {item.get('class_name', 'N/A')}")
            print(f"      Image: {item.get('image_path', 'N/A')}")
            print(f"      Class ID: {item.get('class_id', 'N/A')}")
            print(f"      Confidence: {item.get('conf', 'N/A')}")
            print(f"      Seq ID: {item.get('seq_id', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"❌ Error during data merging: {e}")
        return

if __name__ == "__main__":
    main()
