#!/usr/bin/env python3
"""
Plot per-class accuracy for specific checkpoints or aggregate results.

This script can:
1. Plot per-class accuracy for a specific checkpoint in accumulative training
2. Plot aggregate per-class accuracy across all checkpoints for zs/upper_bound experiments
3. Support both training and test accuracies

Usage examples:
    # Plot specific checkpoint in accumulative training
    python plot_checkpoint_accuracy.py --experiment_path log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name --checkpoint ckp_1 --metrics_dir extracted_metrics
    
    # Plot aggregate results for zs/upper_bound
    python plot_checkpoint_accuracy.py --experiment_path log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name --metrics_dir extracted_metrics --aggregate
    
    # Plot all individual checkpoints for accumulative training
    python plot_checkpoint_accuracy.py --experiment_path log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name --metrics_dir extracted_metrics --all_checkpoints
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

def find_metrics_files(experiment_path, metrics_dir, checkpoint=None):
    """Find relevant metrics files for an experiment."""
    exp_name = os.path.basename(experiment_path)
    
    if checkpoint:
        # Look for specific checkpoint
        pattern = f"{exp_name}_{checkpoint}_per_class_metrics.json"
        files = glob.glob(os.path.join(metrics_dir, pattern))
    else:
        # Look for all checkpoints for this experiment
        pattern = f"{exp_name}_*_per_class_metrics.json"
        files = glob.glob(os.path.join(metrics_dir, pattern))
    
    return sorted(files)

def load_metrics_from_file(file_path):
    """Load metrics from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_single_checkpoint(experiment_path, checkpoint, metrics_dir, output_dir=None):
    """Plot per-class accuracy for a single checkpoint."""
    exp_name = os.path.basename(experiment_path)
    
    # Find metrics file
    metrics_files = find_metrics_files(experiment_path, metrics_dir, checkpoint)
    
    if not metrics_files:
        print(f"No metrics file found for checkpoint {checkpoint}")
        return
    
    metrics = load_metrics_from_file(metrics_files[0])
    
    # Extract data
    class_names = metrics.get('class_names', [f'Class {i}' for i in range(len(metrics['per_class_accuracy']))])
    per_class_acc = metrics['per_class_accuracy']
    overall_acc = metrics['overall_accuracy']
    balanced_acc = metrics['balanced_accuracy']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(class_names))
    bars = ax.bar(x_pos, per_class_acc, alpha=0.7, color='steelblue')
    
    # Add overall and balanced accuracy lines
    ax.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.8, 
               label=f'Overall Accuracy: {overall_acc:.3f}')
    ax.axhline(y=balanced_acc, color='green', linestyle='--', alpha=0.8,
               label=f'Balanced Accuracy: {balanced_acc:.3f}')
    
    # Customize plot
    ax.set_xlabel('Classes')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Per-Class Accuracy - {exp_name}\nCheckpoint: {checkpoint}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{exp_name}_{checkpoint}_per_class_accuracy.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_file}")
    
    plt.show()

def plot_aggregate_checkpoints(experiment_path, metrics_dir, output_dir=None):
    """Plot aggregate per-class accuracy across all checkpoints."""
    exp_name = os.path.basename(experiment_path)
    
    # Find all metrics files
    metrics_files = find_metrics_files(experiment_path, metrics_dir)
    
    if not metrics_files:
        print(f"No metrics files found for experiment {exp_name}")
        return
    
    # Load all metrics
    all_metrics = {}
    for file_path in metrics_files:
        # Extract checkpoint name from filename
        filename = os.path.basename(file_path)
        checkpoint = filename.replace(f'{exp_name}_', '').replace('_per_class_metrics.json', '')
        all_metrics[checkpoint] = load_metrics_from_file(file_path)
    
    if not all_metrics:
        print("No metrics loaded")
        return
    
    # Get class names from first checkpoint
    first_metrics = list(all_metrics.values())[0]
    class_names = first_metrics.get('class_names', [f'Class {i}' for i in range(len(first_metrics['per_class_accuracy']))])
    
    # Aggregate per-class accuracies
    all_per_class = []
    all_overall = []
    all_balanced = []
    checkpoint_names = []
    
    for checkpoint, metrics in sorted(all_metrics.items()):
        all_per_class.append(metrics['per_class_accuracy'])
        all_overall.append(metrics['overall_accuracy'])
        all_balanced.append(metrics['balanced_accuracy'])
        checkpoint_names.append(checkpoint)
    
    # Convert to numpy arrays
    per_class_matrix = np.array(all_per_class)  # Shape: (n_checkpoints, n_classes)
    
    # Calculate statistics
    mean_per_class = np.mean(per_class_matrix, axis=0)
    std_per_class = np.std(per_class_matrix, axis=0)
    max_per_class = np.max(per_class_matrix, axis=0)
    min_per_class = np.min(per_class_matrix, axis=0)
    
    mean_overall = np.mean(all_overall)
    mean_balanced = np.mean(all_balanced)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Mean per-class accuracy with error bars
    x_pos = np.arange(len(class_names))
    bars = ax1.bar(x_pos, mean_per_class, yerr=std_per_class, alpha=0.7, 
                   color='steelblue', capsize=5)
    
    # Add overall and balanced accuracy lines
    ax1.axhline(y=mean_overall, color='red', linestyle='--', alpha=0.8,
                label=f'Mean Overall Accuracy: {mean_overall:.3f}')
    ax1.axhline(y=mean_balanced, color='green', linestyle='--', alpha=0.8,
                label=f'Mean Balanced Accuracy: {mean_balanced:.3f}')
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Mean Per-Class Accuracy (±std) - {exp_name}\nAggregated over {len(checkpoint_names)} checkpoints')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_per_class[i] + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Per-class accuracy across checkpoints (heatmap style)
    im = ax2.imshow(per_class_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Checkpoints')
    ax2.set_title(f'Per-Class Accuracy Across Checkpoints - {exp_name}')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_yticks(range(len(checkpoint_names)))
    ax2.set_yticklabels(checkpoint_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy')
    
    # Add text annotations to heatmap
    for i in range(len(checkpoint_names)):
        for j in range(len(class_names)):
            text = ax2.text(j, i, f'{per_class_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{exp_name}_aggregate_per_class_accuracy.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Aggregate plot saved: {output_file}")
    
    plt.show()
    
    # Also save summary statistics
    summary = {
        'experiment': exp_name,
        'num_checkpoints': len(checkpoint_names),
        'checkpoints': checkpoint_names,
        'class_names': class_names,
        'mean_per_class_accuracy': mean_per_class.tolist(),
        'std_per_class_accuracy': std_per_class.tolist(),
        'max_per_class_accuracy': max_per_class.tolist(),
        'min_per_class_accuracy': min_per_class.tolist(),
        'mean_overall_accuracy': mean_overall,
        'mean_balanced_accuracy': mean_balanced,
        'all_overall_accuracies': all_overall,
        'all_balanced_accuracies': all_balanced
    }
    
    summary_file = os.path.join(output_dir, f'{exp_name}_aggregate_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_file}")

def plot_all_individual_checkpoints(experiment_path, metrics_dir, output_dir=None):
    """Plot individual per-class accuracy for each checkpoint."""
    exp_name = os.path.basename(experiment_path)
    
    # Find all metrics files
    metrics_files = find_metrics_files(experiment_path, metrics_dir)
    
    if not metrics_files:
        print(f"No metrics files found for experiment {exp_name}")
        return
    
    for file_path in metrics_files:
        # Extract checkpoint name from filename
        filename = os.path.basename(file_path)
        checkpoint = filename.replace(f'{exp_name}_', '').replace('_per_class_metrics.json', '')
        
        print(f"Plotting checkpoint: {checkpoint}")
        plot_single_checkpoint(experiment_path, checkpoint, metrics_dir, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Plot per-class accuracy for specific checkpoints or aggregate results')
    parser.add_argument('--experiment_path', type=str, required=True,
                       help='Path to the specific experiment directory')
    parser.add_argument('--checkpoint', type=str,
                       help='Specific checkpoint to plot (e.g., ckp_1, ckp_2)')
    parser.add_argument('--metrics_dir', type=str, default='extracted_metrics',
                       help='Directory containing extracted metrics files')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for plots')
    parser.add_argument('--aggregate', action='store_true',
                       help='Generate aggregate plot across all checkpoints')
    parser.add_argument('--all_checkpoints', action='store_true',
                       help='Generate individual plots for all checkpoints')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiment_path):
        print(f"Error: Experiment path does not exist: {args.experiment_path}")
        return
    
    if not os.path.exists(args.metrics_dir):
        print(f"Error: Metrics directory does not exist: {args.metrics_dir}")
        return
    
    if args.checkpoint:
        # Plot specific checkpoint
        plot_single_checkpoint(args.experiment_path, args.checkpoint, args.metrics_dir, args.output_dir)
    elif args.aggregate:
        # Plot aggregate across all checkpoints
        plot_aggregate_checkpoints(args.experiment_path, args.metrics_dir, args.output_dir)
    elif args.all_checkpoints:
        # Plot all individual checkpoints
        plot_all_individual_checkpoints(args.experiment_path, args.metrics_dir, args.output_dir)
    else:
        print("Please specify either --checkpoint, --aggregate, or --all_checkpoints")
        print("Use --help for more information")

if __name__ == '__main__':
    main()
