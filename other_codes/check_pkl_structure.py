#!/usr/bin/env python3
"""
Check the structure of prediction pkl files to understand train/test data format.
"""

import pickle
import numpy as np
import os
import glob

def check_pkl_file(pkl_file):
    """Check the structure of a single pkl file."""
    print(f"\n=== Checking: {pkl_file} ===")
    
    if not os.path.exists(pkl_file):
        print(f"File does not exist: {pkl_file}")
        return
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        
        if isinstance(data, (list, tuple)):
            print(f"Number of items: {len(data)}")
            for i, item in enumerate(data):
                print(f"  Item {i}:")
                print(f"    Type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"    Shape: {item.shape}")
                    print(f"    Sample values: {item[:5] if len(item) > 0 else 'Empty'}")
                elif isinstance(item, (list, tuple)):
                    print(f"    Length: {len(item)}")
                    if len(item) > 0:
                        print(f"    First element type: {type(item[0])}")
                        if hasattr(item[0], 'shape'):
                            print(f"    First element shape: {item[0].shape}")
                else:
                    print(f"    Value: {item}")
        
        elif hasattr(data, 'shape'):
            print(f"Data shape: {data.shape}")
            print(f"Sample values: {data[:5] if len(data) > 0 else 'Empty'}")
        
        elif isinstance(data, dict):
            print(f"Dictionary with keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: type={type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                    print(f"    Sample: {value[:5] if len(value) > 0 else 'Empty'}")
        
        else:
            print(f"Unrecognized data type: {type(data)}")
            print(f"Data: {data}")
            
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")

def main():
    # Check files from different experiment types
    
    # Files to check
    test_files = [
        # ZS experiment
        "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name/full/log/ckp_1_preds.pkl",
        
        # Accumulative experiment (if exists)
        "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/na_lebec_CA-22/ce/accumulative-scratch/bioclip2_2025-06-27-23-04-46_common_name/full/log/ckp_1_preds.pkl",
    ]
    
    # Also search for any available pkl files
    print("=== Searching for available prediction files ===")
    
    search_patterns = [
        "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/*/*/*/*/full/log/*_preds.pkl",
        "/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/*/*/*/*/*/full/log/*_preds.pkl",
    ]
    
    found_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    found_files = list(set(found_files))  # Remove duplicates
    print(f"Found {len(found_files)} prediction files")
    
    # Check first few files
    for i, file_path in enumerate(found_files[:3]):  # Check first 3 files
        check_pkl_file(file_path)
    
    # Check specific test files
    for file_path in test_files:
        check_pkl_file(file_path)

if __name__ == '__main__':
    main()
