#!/usr/bin/env python3
"""
Plot per-class accuracy comparison from saved JSON metrics.

Usage:
    python plot_per_class_accuracy.py --dataset ENO_C05 --methods focal_accumulative-scratch,ce_accumulative-scratch
"""

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics_from_json(json_path):
    """Load metrics from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def find_metrics_files(log_dir, dataset, method_pattern):
    """Find all per-class metrics JSON files for a given dataset and method."""
    # First try the new format (direct saving)
    pattern1 = os.path.join(log_dir, f"**/{dataset}/**/{method_pattern}/**/*_per_class_metrics.json")
    files1 = glob.glob(pattern1, recursive=True)
    
    # Also try the extracted format
    pattern2 = os.path.join(log_dir, f"**/*{dataset}*{method_pattern}*_per_class_metrics.json")
    files2 = glob.glob(pattern2, recursive=True)
    
    # Combine and deduplicate
    all_files = list(set(files1 + files2))
    
    return all_files

def parse_method_name(filepath):
    """Extract method name from file path."""
    parts = filepath.split(os.sep)
    # Typical path: log/pipeline/ENO_C05/focal/accumulative-scratch/ckp_per_class_metrics.json
    if len(parts) >= 3:
        # Extract loss_type and cl_method
        for i, part in enumerate(parts):
            if part in ['focal', 'ce', 'cb-focal', 'ldam', 'cdt']:
                loss_type = part
                if i + 1 < len(parts):
                    cl_method = parts[i + 1]
                    return f"{loss_type}_{cl_method}"
    return "unknown"

def plot_per_class_comparison(datasets_metrics, save_path=None, figsize=(12, 8)):
    """Plot per-class accuracy comparison."""
    
    fig, axes = plt.subplots(1, len(datasets_metrics), figsize=figsize, squeeze=False)
    if len(datasets_metrics) == 1:
        axes = [axes[0]]
    else:
        axes = axes[0]
    
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    for dataset_idx, (dataset_name, methods_data) in enumerate(datasets_metrics.items()):
        ax = axes[dataset_idx]
        
        for method_idx, (method_name, metrics_list) in enumerate(methods_data.items()):
            if not metrics_list:
                continue
                
            # Use the latest checkpoint or average across checkpoints
            metrics = metrics_list[-1]  # Use last checkpoint
            
            n_classes = len(metrics['per_class_accuracy'])
            class_indices = list(range(1, n_classes + 1))
            per_class_acc = [acc * 100 for acc in metrics['per_class_accuracy']]
            
            # Plot per-class accuracy
            ax.plot(class_indices, per_class_acc, 'o-', 
                   label=method_name.replace('_', ' + '), 
                   color=colors[method_idx % len(colors)],
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('Class Index', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Per-class Accuracy - {dataset_name}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot per-class accuracy comparison')
    parser.add_argument('--log_dir', type=str, default='log/pipeline',
                       help='Base directory containing logs')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., ENO_C05)')
    parser.add_argument('--methods', type=str, 
                       help='Comma-separated method patterns (e.g., focal,ce)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for the plot')
    parser.add_argument('--figsize', type=str, default='12,8',
                       help='Figure size as width,height')
    
    args = parser.parse_args()
    
    # Parse figure size
    figsize = tuple(map(int, args.figsize.split(',')))
    
    # Collect data
    datasets_metrics = {}
    
    if args.methods:
        method_patterns = [m.strip() for m in args.methods.split(',')]
    else:
        # Auto-discover methods
        method_patterns = ['*']
    
    print(f"Looking for dataset: {args.dataset}")
    print(f"Method patterns: {method_patterns}")
    
    datasets_metrics[args.dataset] = {}
    
    for pattern in method_patterns:
        files = find_metrics_files(args.log_dir, args.dataset, pattern)
        print(f"Found {len(files)} files for pattern '{pattern}'")
        
        for file_path in files:
            print(f"Processing: {file_path}")
            try:
                metrics = load_metrics_from_json(file_path)
                method_name = metrics.get('method', parse_method_name(file_path))
                
                if method_name not in datasets_metrics[args.dataset]:
                    datasets_metrics[args.dataset][method_name] = []
                
                datasets_metrics[args.dataset][method_name].append(metrics)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Print summary
    print(f"\nDatasets and methods found:")
    for dataset, methods in datasets_metrics.items():
        print(f"  {dataset}:")
        for method, metrics_list in methods.items():
            print(f"    {method}: {len(metrics_list)} checkpoints")
    
    if not any(datasets_metrics.values()):
        print("No metrics data found!")
        return
    
    # Generate output path if not provided
    if args.output is None:
        args.output = f"per_class_accuracy_{args.dataset}.png"
    
    # Plot
    plot_per_class_comparison(datasets_metrics, args.output, figsize)

if __name__ == '__main__':
    main()
