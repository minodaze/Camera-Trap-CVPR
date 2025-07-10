#!/usr/bin/env python3
"""
Focal Loss Hyperparameter Configuration for ICICLE-Benchmark

This script provides optimized hyperparameters for different loss functions
based on dataset characteristics and class imbalance ratios.
"""

import numpy as np
import json
import os

def calculate_imbalance_ratio(samples_per_cls):
    """Calculate class imbalance ratio"""
    max_samples = max(samples_per_cls)
    min_samples = min(samples_per_cls)
    return max_samples / min_samples

def get_focal_hyperparameters(samples_per_cls, loss_type='cb-focal'):
    """
    Get optimized hyperparameters based on class distribution
    
    Args:
        samples_per_cls: List of sample counts per class
        loss_type: Type of loss function
    
    Returns:
        Dictionary with optimized hyperparameters
    """
    imbalance_ratio = calculate_imbalance_ratio(samples_per_cls)
    
    if loss_type == 'cb-focal':
        if imbalance_ratio > 1000:
            # Extremely imbalanced (e.g., rare species datasets)
            return {
                'beta': 0.99999,   # Very strong class balancing
                'gamma': 3.0,      # Strong focusing on hard examples
                'comment': f'Extreme imbalance (ratio: {imbalance_ratio:.1f})'
            }
        elif imbalance_ratio > 100:
            # Highly imbalanced (typical for species datasets)
            return {
                'beta': 0.9999,    # Strong class balancing
                'gamma': 2.0,      # Standard focusing
                'comment': f'High imbalance (ratio: {imbalance_ratio:.1f})'
            }
        elif imbalance_ratio > 10:
            # Moderately imbalanced
            return {
                'beta': 0.999,     # Moderate class balancing
                'gamma': 1.5,      # Light focusing
                'comment': f'Moderate imbalance (ratio: {imbalance_ratio:.1f})'
            }
        else:
            # Relatively balanced
            return {
                'beta': 0.99,      # Light class balancing
                'gamma': 1.0,      # Minimal focusing
                'comment': f'Low imbalance (ratio: {imbalance_ratio:.1f})'
            }
    
    elif loss_type == 'focal':
        # Pure focal loss without class balancing
        if imbalance_ratio > 100:
            return {
                'gamma': 3.0,      # Strong focusing for highly imbalanced data
                'comment': f'High imbalance (ratio: {imbalance_ratio:.1f})'
            }
        elif imbalance_ratio > 10:
            return {
                'gamma': 2.0,      # Standard focusing
                'comment': f'Moderate imbalance (ratio: {imbalance_ratio:.1f})'
            }
        else:
            return {
                'gamma': 1.0,      # Light focusing
                'comment': f'Low imbalance (ratio: {imbalance_ratio:.1f})'
            }
    
    elif loss_type == 'cdt':
        # CDT loss parameters
        if imbalance_ratio > 100:
            return {
                'gamma': 0.5,      # Strong temperature scaling
                'comment': f'High imbalance (ratio: {imbalance_ratio:.1f})'
            }
        elif imbalance_ratio > 10:
            return {
                'gamma': 0.3,      # Moderate temperature scaling (current setting)
                'comment': f'Moderate imbalance (ratio: {imbalance_ratio:.1f})'
            }
        else:
            return {
                'gamma': 0.1,      # Light temperature scaling
                'comment': f'Low imbalance (ratio: {imbalance_ratio:.1f})'
            }

def load_dataset_stats(train_json_path):
    """Load dataset and calculate class distribution"""
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    
    # Count samples per class
    class_counts = {}
    for item in data:
        class_name = item['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return list(class_counts.values()), list(class_counts.keys())

def main():
    """Main function to demonstrate parameter selection"""
    
    # Example datasets with different imbalance characteristics
    datasets = {
        'na_lebec_CA-22': '/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/na/na_lebec_CA-22_common_name/train.json',
        'ENO_B06': '/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/ENO/ENO_B06/train.json',
        'MAD_MAD05': '/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/data/MAD/MAD_MAD05_new/train.json'
    }
    
    print("🔥 FOCAL LOSS HYPERPARAMETER RECOMMENDATIONS 🔥")
    print("=" * 60)
    
    for dataset_name, json_path in datasets.items():
        if os.path.exists(json_path):
            try:
                samples_per_cls, class_names = load_dataset_stats(json_path)
                imbalance_ratio = calculate_imbalance_ratio(samples_per_cls)
                
                print(f"\n📊 Dataset: {dataset_name}")
                print(f"   Classes: {len(class_names)}")
                print(f"   Total samples: {sum(samples_per_cls)}")
                print(f"   Imbalance ratio: {imbalance_ratio:.1f}")
                print(f"   Max samples: {max(samples_per_cls)}")
                print(f"   Min samples: {min(samples_per_cls)}")
                
                # Get recommendations for different loss types
                cb_focal_params = get_focal_hyperparameters(samples_per_cls, 'cb-focal')
                cdt_params = get_focal_hyperparameters(samples_per_cls, 'cdt')
                
                print(f"\n   🎯 RECOMMENDED PARAMETERS:")
                print(f"   CB-Focal Loss:")
                print(f"     beta: {cb_focal_params['beta']}")
                print(f"     gamma: {cb_focal_params['gamma']}")
                print(f"     reason: {cb_focal_params['comment']}")
                
                print(f"   CDT Loss:")
                print(f"     gamma: {cdt_params['gamma']}")
                print(f"     reason: {cdt_params['comment']}")
                
            except Exception as e:
                print(f"\n❌ Error processing {dataset_name}: {e}")
        else:
            print(f"\n⚠️  Dataset {dataset_name} not found at {json_path}")
    
    print(f"\n" + "=" * 60)
    print("💡 GENERAL RECOMMENDATIONS:")
    print("   • For biological species data (long-tail): cb-focal with β=0.9999, γ=2.0")
    print("   • For extremely rare species: cb-focal with β=0.99999, γ=3.0")
    print("   • For temperature-based approach: cdt with γ=0.3")
    print("   • Start with conservative values and tune based on validation performance")
    print("=" * 60)

if __name__ == "__main__":
    main()
