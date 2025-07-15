#!/usr/bin/env python3
"""
Plot per-class accuracy for specific checkpoints or aggregated results.

This script can:
1. For accumulative training: Plot train/test per-class accuracy for a specific checkpoint
2. For zs/upper_bound: Plot aggregated train/test per-class accuracy across all checkpoints

Usage examples:
    # Plot specific checkpoint for accumulative training
    python plot_checkpoint_accuracy.py --log_path log/pipeline/ENO_C05/cdt/accumulative-scratch/bioclip2_2025-07-06-22-13-54_scientific_name --checkpoint ckp_5 --mode accumulative
    
    # Plot aggregated results for zs/upper_bound
    python plot_checkpoint_accuracy.py --log_path log/pipeline/ENO_C05/cdt/zs/bioclip2_2025-07-06-22-13-54_scientific_name --mode aggregate
"""

import argparse
import os
import pickle
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ruamel.yaml as yaml

def load_class_names_from_config(config_path):
    """Load class names from the experiment configuration."""
    try:
        with open(config_path, 'r') as f:
            yml = yaml.YAML(typ='rt')
            config = yml.load(f)
        return config.get('class_names', [])
    except Exception as e:
        print(f"Warning: Could not load class names from {config_path}: {e}")
        return None

def find_config_file(experiment_dir):
    """Find the configuration file in the experiment directory."""
    # Look for args.yaml first
    args_yaml = os.path.join(experiment_dir, 'args.yaml')
    if os.path.exists(args_yaml):
        return args_yaml
    
    # Look for other yaml files
    yaml_files = glob.glob(os.path.join(experiment_dir, '*.yaml'))
    if yaml_files:
        return yaml_files[0]
    
    return None

def compute_per_class_metrics(preds_arr, labels_arr, n_classes, class_names):
    """Compute per-class accuracy and other metrics."""
    # Calculate per-class accuracy
    acc_per_class = []
    samples_per_class = []
    
    for i in range(n_classes):
        mask = labels_arr == i
        n_samples = mask.sum()
        samples_per_class.append(int(n_samples))
        
        if n_samples == 0:
            acc_per_class.append(0.0)
        else:
            acc_per_class.append(float((preds_arr[mask] == labels_arr[mask]).mean()))
    
    # Calculate overall metrics
    acc = float((preds_arr == labels_arr).mean())
    balanced_acc = float(np.array(acc_per_class).mean())
    
    return {
        'overall_accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'per_class_accuracy': acc_per_class,
        'samples_per_class': samples_per_class,
        'class_names': class_names if isinstance(class_names, list) else list(class_names),
    }

def load_checkpoint_predictions(log_path, checkpoint_name):
    """Load predictions for a specific checkpoint."""
    # Look for both train and test predictions
    train_pred_file = os.path.join(log_path, f"{checkpoint_name}_train_preds.pkl")
    test_pred_file = os.path.join(log_path, f"{checkpoint_name}_test_preds.pkl")
    
    results = {}
    
    if os.path.exists(train_pred_file):
        try:
            with open(train_pred_file, 'rb') as f:
                preds_arr, labels_arr = pickle.load(f)
            results['train'] = (preds_arr, labels_arr)
            print(f"Loaded train predictions from {train_pred_file}")
        except Exception as e:
            print(f"Error loading train predictions: {e}")
    
    if os.path.exists(test_pred_file):
        try:
            with open(test_pred_file, 'rb') as f:
                preds_arr, labels_arr = pickle.load(f)
            results['test'] = (preds_arr, labels_arr)
            print(f"Loaded test predictions from {test_pred_file}")
        except Exception as e:
            print(f"Error loading test predictions: {e}")
    
    # If specific train/test files don't exist, try generic prediction file
    if not results:
        generic_pred_file = os.path.join(log_path, f"{checkpoint_name}_preds.pkl")
        if os.path.exists(generic_pred_file):
            try:
                with open(generic_pred_file, 'rb') as f:
                    preds_arr, labels_arr = pickle.load(f)
                results['test'] = (preds_arr, labels_arr)  # Assume it's test data
                print(f"Loaded predictions from {generic_pred_file} (assuming test data)")
            except Exception as e:
                print(f"Error loading predictions: {e}")
    
    return results

def load_all_checkpoints(log_path):
    """Load predictions from all checkpoints in the directory."""
    pred_files = glob.glob(os.path.join(log_path, '*_preds.pkl'))
    
    all_results = {'train': [], 'test': []}
    
    for pred_file in sorted(pred_files):
        try:
            # Extract checkpoint name and type
            filename = os.path.basename(pred_file).replace('_preds.pkl', '')
            
            # Determine if it's train or test
            if filename.endswith('_train'):
                data_type = 'train'
                ckp_name = filename.replace('_train', '')
            elif filename.endswith('_test'):
                data_type = 'test'
                ckp_name = filename.replace('_test', '')
            else:
                data_type = 'test'  # Default to test
                ckp_name = filename
            
            # Load predictions
            with open(pred_file, 'rb') as f:
                preds_arr, labels_arr = pickle.load(f)
            
            all_results[data_type].append({
                'checkpoint': ckp_name,
                'predictions': preds_arr,
                'labels': labels_arr
            })
            
            print(f"Loaded {data_type} data for checkpoint {ckp_name}")
            
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
            continue
    
    return all_results

def aggregate_predictions(data_list):
    """Aggregate predictions from multiple checkpoints."""
    if not data_list:
        return None, None
    
    all_preds = []
    all_labels = []
    
    for data in data_list:
        all_preds.append(data['predictions'])
        all_labels.append(data['labels'])
    
    # Concatenate all predictions and labels
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    
    return preds_arr, labels_arr

def plot_per_class_accuracy(metrics_dict, title, output_path=None):
    """Plot per-class accuracy comparison."""
    class_names = metrics_dict[list(metrics_dict.keys())[0]]['class_names']
    n_classes = len(class_names)
    
    # Prepare data for plotting
    data_types = list(metrics_dict.keys())
    x = np.arange(n_classes)
    width = 0.35 if len(data_types) == 2 else 0.8
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n_classes), 10))
    
    # Plot 1: Per-class accuracy comparison
    if len(data_types) == 2:
        ax1.bar(x - width/2, metrics_dict[data_types[0]]['per_class_accuracy'], 
                width, label=data_types[0], alpha=0.8)
        ax1.bar(x + width/2, metrics_dict[data_types[1]]['per_class_accuracy'], 
                width, label=data_types[1], alpha=0.8)
    else:
        ax1.bar(x, metrics_dict[data_types[0]]['per_class_accuracy'], 
                width, label=data_types[0], alpha=0.8)
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{title} - Per-Class Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add accuracy values on bars
    for i, data_type in enumerate(data_types):
        accuracies = metrics_dict[data_type]['per_class_accuracy']
        offset = (-width/2 if i == 0 and len(data_types) == 2 else 
                 width/2 if i == 1 and len(data_types) == 2 else 0)
        for j, acc in enumerate(accuracies):
            ax1.text(j + offset, acc + 0.01, f'{acc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Sample distribution
    samples_data = metrics_dict[list(metrics_dict.keys())[0]]['samples_per_class']
    bars = ax2.bar(x, samples_data, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title(f'{title} - Sample Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add sample counts on bars
    for i, (bar, count) in enumerate(zip(bars, samples_data)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(samples_data)*0.01,
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n{title} Summary:")
    for data_type, metrics in metrics_dict.items():
        print(f"{data_type.capitalize()}:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Per-class Accuracy: {[f'{acc:.3f}' for acc in metrics['per_class_accuracy']]}")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot per-class accuracy for specific checkpoints')
    parser.add_argument('--log_path', type=str, required=True,
                       help='Path to experiment log directory')
    parser.add_argument('--checkpoint', type=str,
                       help='Specific checkpoint name (e.g., ckp_5) for accumulative mode')
    parser.add_argument('--mode', type=str, choices=['accumulative', 'aggregate'], required=True,
                       help='accumulative: plot specific checkpoint; aggregate: plot all checkpoints combined')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--class_names', type=str, nargs='+',
                       help='Manually specify class names if not found in config')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.log_path):
        print(f"Error: Log path does not exist: {args.log_path}")
        return
    
    if args.mode == 'accumulative' and not args.checkpoint:
        print("Error: --checkpoint is required for accumulative mode")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load class names
    class_names = args.class_names
    if class_names is None:
        config_file = find_config_file(args.log_path)
        if config_file:
            class_names = load_class_names_from_config(config_file)
        
        if class_names is None:
            print("Error: Could not find class names. Please specify with --class_names")
            return
    
    n_classes = len(class_names)
    print(f"Found {n_classes} classes: {class_names}")
    
    # Extract experiment info from path
    path_parts = args.log_path.split(os.sep)
    dataset = method = loss_type = "unknown"
    for i, part in enumerate(path_parts):
        if part in ['ENO_C05', 'APN_K024', 'MAD_MAD05', 'serengeti', 'ENO_B06', 'PLN_D01']:
            dataset = part
            if i + 1 < len(path_parts):
                loss_type = path_parts[i + 1]
            if i + 2 < len(path_parts):
                method = path_parts[i + 2]
            break
    
    experiment_name = f"{dataset}_{loss_type}_{method}"
    
    if args.mode == 'accumulative':
        print(f"Loading checkpoint {args.checkpoint} for accumulative training...")
        
        # Load specific checkpoint
        checkpoint_data = load_checkpoint_predictions(args.log_path, args.checkpoint)
        
        if not checkpoint_data:
            print(f"Error: No prediction data found for checkpoint {args.checkpoint}")
            return
        
        # Compute metrics for each data type
        metrics_dict = {}
        for data_type, (preds_arr, labels_arr) in checkpoint_data.items():
            metrics = compute_per_class_metrics(preds_arr, labels_arr, n_classes, class_names)
            metrics_dict[data_type] = metrics
        
        # Plot results
        title = f"{experiment_name} - Checkpoint {args.checkpoint}"
        output_path = os.path.join(args.output_dir, f"{experiment_name}_{args.checkpoint}_per_class.png")
        plot_per_class_accuracy(metrics_dict, title, output_path)
        
    elif args.mode == 'aggregate':
        print(f"Loading all checkpoints for aggregation...")
        
        # Load all checkpoints
        all_data = load_all_checkpoints(args.log_path)
        
        # Aggregate predictions
        metrics_dict = {}
        for data_type in ['train', 'test']:
            if all_data[data_type]:
                preds_arr, labels_arr = aggregate_predictions(all_data[data_type])
                if preds_arr is not None:
                    metrics = compute_per_class_metrics(preds_arr, labels_arr, n_classes, class_names)
                    metrics_dict[data_type] = metrics
                    print(f"Aggregated {len(all_data[data_type])} {data_type} checkpoints")
        
        if not metrics_dict:
            print("Error: No prediction data found for aggregation")
            return
        
        # Plot results
        title = f"{experiment_name} - Aggregated Results"
        output_path = os.path.join(args.output_dir, f"{experiment_name}_aggregated_per_class.png")
        plot_per_class_accuracy(metrics_dict, title, output_path)

if __name__ == '__main__':
    main()
