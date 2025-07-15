#!/usr/bin/env python3
"""
Extract per-class metrics from previously completed experiments.

This script reads the saved prediction files (preds.pkl) and computes per-class accuracy
for experiments that were run before the per-class metrics saving was implemented.

The script can handle checkpoint files in various locations:
- Direct in experiment directory: experiment_path/ckp_1_preds.pkl
- In log subdirectory: experiment_path/log/ckp_1_preds.pkl  
- In full/log subdirectory: experiment_path/full/log/ckp_1_preds.pkl

Usage examples:
    # Extract specific checkpoint
    python extract_per_class_from_completed.py --experiment_path log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name --checkpoint ckp_1 --plot
    
    # Extract all checkpoints with aggregate plot
    python extract_per_class_from_completed.py --experiment_path log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name --plot --plot_type aggregate
"""

import argparse
import os
import pickle
import json
import glob
import numpy as np
from pathlib import Path
import ruamel.yaml as yaml

def load_class_names_from_config(config_path):
    """Load class names from the experiment configuration."""
    try:
        with open(config_path, 'r') as f:
            yml = yaml.YAML(typ='rt')
            config = yml.load(f)
        return config.get('class_names', [])
    except:
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

def process_experiment_directory(exp_dir, class_names=None):
    """Process a single experiment directory and extract per-class metrics."""
    print(f"Processing: {exp_dir}")
    
    # Try to get class names from config if not provided
    if class_names is None:
        config_file = find_config_file(exp_dir)
        if config_file:
            class_names = load_class_names_from_config(config_file)
        
        if class_names is None:
            print(f"  Warning: Could not find class names for {exp_dir}")
            return None
    
    n_classes = len(class_names)
    
    # Find all prediction files in current directory and subdirectories
    pred_files = []
    for pattern in ['*_preds.pkl', 'log/*_preds.pkl', 'full/log/*_preds.pkl']:
        pred_files.extend(glob.glob(os.path.join(exp_dir, pattern)))
    
    if not pred_files:
        print(f"  No prediction files found in {exp_dir}")
        return None
    
    print(f"  Found {len(pred_files)} prediction files")
    
    # Process each checkpoint
    results = {}
    
    for pred_file in sorted(pred_files):
        try:
            # Extract checkpoint name
            ckp_name = os.path.basename(pred_file).replace('_preds.pkl', '')
            
            # Load predictions
            with open(pred_file, 'rb') as f:
                preds_arr, labels_arr = pickle.load(f)
            
            # Compute per-class metrics
            metrics = compute_per_class_metrics(preds_arr, labels_arr, n_classes, class_names)
            metrics['checkpoint'] = ckp_name
            
            # Determine method name from directory structure
            dir_parts = exp_dir.split(os.sep)
            if len(dir_parts) >= 3:
                # Try to extract dataset, loss_type, method from path
                dataset = loss_type = method = "unknown"
                for i, part in enumerate(dir_parts):
                    if part in ['ENO_C05', 'ENO_E06', 'APN_K024', 'MAD_MAD05', 'serengeti']:
                        dataset = part
                        if i + 1 < len(dir_parts):
                            loss_type = dir_parts[i + 1]
                        if i + 2 < len(dir_parts):
                            method = dir_parts[i + 2]
                        break
                metrics['method'] = f"{loss_type}_{method}"
                metrics['dataset'] = dataset
            else:
                metrics['method'] = f"unknown_{os.path.basename(exp_dir)}"
            
            results[ckp_name] = metrics
            
            print(f"    {ckp_name}: acc={metrics['overall_accuracy']:.4f}, balanced_acc={metrics['balanced_accuracy']:.4f}")
            
        except Exception as e:
            print(f"    Error processing {pred_file}: {e}")
            continue
    
    return results

def save_extracted_metrics(results, output_dir):
    """Save extracted metrics to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for exp_name, exp_results in results.items():
        for ckp_name, metrics in exp_results.items():
            # Create filename similar to the new format
            output_file = os.path.join(output_dir, f"{exp_name}_{ckp_name}_per_class_metrics.json")
            
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Saved: {output_file}")

def get_class_names(experiment_path, manual_class_names=None):
    """Get class names from experiment configuration or manual input."""
    if manual_class_names:
        return manual_class_names
    
    # Try to find class names from config files
    config_dirs = [
        experiment_path,
        os.path.dirname(experiment_path),
        os.path.join(os.path.dirname(experiment_path), '..'),
        'config'
    ]
    
    for config_dir in config_dirs:
        # Look for yaml files that might contain class names
        for yaml_file in glob.glob(os.path.join(config_dir, '*.yaml')):
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'data_config' in config and 'class_names' in config['data_config']:
                        return config['data_config']['class_names']
            except Exception:
                continue
        
        # Look for common_name_lookup.json
        lookup_file = os.path.join(config_dir, 'common_name_lookup.json')
        if os.path.exists(lookup_file):
            try:
                with open(lookup_file, 'r') as f:
                    lookup = json.load(f)
                    # Extract unique class names
                    class_names = list(set(lookup.values()))
                    return sorted(class_names)
            except Exception:
                continue
    
    # Default class names if nothing found
    print("Warning: Could not find class names in config. Using default numeric labels.")
    return None

def extract_single_checkpoint(experiment_path, checkpoint, class_names, output_dir):
    """Extract metrics for a single checkpoint from an experiment."""
    print(f"Extracting single checkpoint: {checkpoint}")
    
    # Look for the checkpoint file in various possible locations
    possible_paths = [
        os.path.join(experiment_path, f"{checkpoint}_preds.pkl"),  # Direct in experiment path
        os.path.join(experiment_path, "log", f"{checkpoint}_preds.pkl"),  # In log subdirectory
        os.path.join(experiment_path, "full", "log", f"{checkpoint}_preds.pkl"),  # In full/log subdirectory
    ]
    
    checkpoint_file = None
    for path in possible_paths:
        if os.path.exists(path):
            checkpoint_file = path
            break
    
    if checkpoint_file is None:
        print(f"Checkpoint file not found for {checkpoint}")
        print("Searched in:")
        for path in possible_paths:
            print(f"  {path}")
        
        # List available checkpoint files
        search_dirs = [
            experiment_path,
            os.path.join(experiment_path, "log"),
            os.path.join(experiment_path, "full", "log")
        ]
        
        print("\nAvailable checkpoint files:")
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                pred_files = glob.glob(os.path.join(search_dir, "*_preds.pkl"))
                if pred_files:
                    print(f"  In {search_dir}:")
                    for f in sorted(pred_files):
                        print(f"    {os.path.basename(f)}")
        return
    
    print(f"Found checkpoint file: {checkpoint_file}")
    
    # Process the checkpoint by processing the directory containing it
    checkpoint_dir = os.path.dirname(checkpoint_file)
    results = process_experiment_directory(checkpoint_dir, class_names)
    
    if results and checkpoint in results:
        # Create experiment name from path
        exp_name = os.path.basename(experiment_path)
        
        # Save the specific checkpoint
        single_result = {exp_name: {checkpoint: results[checkpoint]}}
        save_extracted_metrics(single_result, output_dir)
        print(f"Extracted checkpoint {checkpoint} from {exp_name}")
    else:
        print(f"Checkpoint {checkpoint} not found in processed results")
        if results:
            print(f"Available checkpoints: {list(results.keys())}")

def extract_all_checkpoints(experiment_path, class_names, output_dir):
    """Extract metrics for all checkpoints from an experiment."""
    print(f"Extracting all checkpoints from: {experiment_path}")
    
    # Look for checkpoints in various possible locations
    search_dirs = [
        experiment_path,
        os.path.join(experiment_path, "log"),
        os.path.join(experiment_path, "full", "log")
    ]
    
    results = None
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            results = process_experiment_directory(search_dir, class_names)
            if results:
                break
    
    if results:
        exp_name = os.path.basename(experiment_path)
        all_results = {exp_name: results}
        save_extracted_metrics(all_results, output_dir)
        print(f"Extracted {len(results)} checkpoints from {exp_name}")
    else:
        print("No checkpoints found")

def plot_single_checkpoint(experiment_path, checkpoint, class_names, output_dir):
    """Generate plot for a single checkpoint."""
    import subprocess
    
    # Find the extracted metrics file
    exp_name = os.path.basename(experiment_path)
    metrics_file = os.path.join(output_dir, f"{exp_name}_{checkpoint}_per_class_metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    # Generate plot using the plotting script
    plot_cmd = [
        'python', 'plot_checkpoint_accuracy.py',
        '--experiment_path', experiment_path,
        '--checkpoint', checkpoint,
        '--metrics_dir', output_dir
    ]
    
    print(f"Running: {' '.join(plot_cmd)}")
    try:
        subprocess.run(plot_cmd, check=True)
        print(f"Plot generated for checkpoint {checkpoint}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating plot: {e}")

def plot_aggregate_checkpoints(experiment_path, class_names, output_dir):
    """Generate aggregate plot for all checkpoints."""
    import subprocess
    
    # Generate plot using the plotting script
    plot_cmd = [
        'python', 'plot_checkpoint_accuracy.py',
        '--experiment_path', experiment_path,
        '--metrics_dir', output_dir,
        '--aggregate'
    ]
    
    print(f"Running: {' '.join(plot_cmd)}")
    try:
        subprocess.run(plot_cmd, check=True)
        print(f"Aggregate plot generated for all checkpoints")
    except subprocess.CalledProcessError as e:
        print(f"Error generating plot: {e}")

def plot_all_individual_checkpoints(experiment_path, class_names, output_dir):
    """Generate individual plots for each checkpoint."""
    results = process_experiment_directory(experiment_path, class_names)
    
    for checkpoint in results.keys():
        plot_single_checkpoint(experiment_path, checkpoint, class_names, output_dir)

def process_experiments(log_dir, dataset_filter, method_filter, class_names, output_dir):
    """Process experiments based on filters (legacy functionality)."""
    # Find experiment directories
    if dataset_filter and method_filter:
        # Specific experiment
        pattern = os.path.join(log_dir, dataset_filter, method_filter, '**/log')
    elif dataset_filter:
        # All methods for a dataset
        pattern = os.path.join(log_dir, dataset_filter, '**/log')
    else:
        # All experiments
        pattern = os.path.join(log_dir, '**/log')
    
    print(f"Searching pattern: {pattern}")
    exp_dirs = glob.glob(pattern, recursive=True)
    
    if not exp_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Process each experiment
    all_results = {}
    
    for exp_dir in sorted(exp_dirs):
        # Create a unique name for this experiment
        rel_path = os.path.relpath(exp_dir, log_dir)
        exp_name = rel_path.replace(os.sep, '_').replace('_log', '')
        
        results = process_experiment_directory(exp_dir, class_names)
        
        if results:
            all_results[exp_name] = results
            print(f"  Processed {len(results)} checkpoints")
        else:
            print(f"  No results extracted")
        
        print()
    
    # Save results
    if all_results:
        save_extracted_metrics(all_results, output_dir)
        
        # Create summary
        summary_file = os.path.join(output_dir, 'extraction_summary.json')
        summary = {
            'total_experiments': len(all_results),
            'experiments': {}
        }
        
        for exp_name, exp_results in all_results.items():
            summary['experiments'][exp_name] = {
                'checkpoints': len(exp_results),
                'latest_accuracy': list(exp_results.values())[-1]['overall_accuracy'] if exp_results else 0,
                'latest_balanced_accuracy': list(exp_results.values())[-1]['balanced_accuracy'] if exp_results else 0
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExtraction complete!")
        print(f"Summary saved to: {summary_file}")
        print(f"Metrics saved to: {output_dir}")
        
    else:
        print("No results were extracted!")

def main():
    parser = argparse.ArgumentParser(description='Extract per-class metrics from completed experiments')
    parser.add_argument('--log_dir', type=str, default='log/pipeline',
                       help='Base directory containing experiment logs')
    parser.add_argument('--dataset', type=str, 
                       help='Dataset name (e.g., ENO_C05). If not specified, process all datasets')
    parser.add_argument('--method', type=str,
                       help='Method pattern (e.g., focal/accumulative-scratch). If not specified, process all methods')
    parser.add_argument('--experiment_path', type=str,
                       help='Direct path to specific experiment directory (e.g., log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name)')
    parser.add_argument('--checkpoint', type=str,
                       help='Specific checkpoint to analyze (e.g., ckp_1, ckp_2). For accumulative training only.')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots after extraction')
    parser.add_argument('--plot_type', type=str, choices=['single_checkpoint', 'aggregate'], default='single_checkpoint',
                       help='Type of plot: single_checkpoint for specific checkpoint, aggregate for all checkpoints combined')
    parser.add_argument('--output_dir', type=str, default='extracted_metrics',
                       help='Output directory for extracted metrics')
    parser.add_argument('--class_names', type=str, nargs='+',
                       help='Manually specify class names if not found in config')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.experiment_path:
        # Process specific experiment path
        if not os.path.exists(args.experiment_path):
            print(f"Error: Experiment path does not exist: {args.experiment_path}")
            return
            
        print(f"Processing specific experiment: {args.experiment_path}")
        class_names = get_class_names(args.experiment_path, args.class_names)
        
        if args.checkpoint:
            # Process specific checkpoint for accumulative training
            extract_single_checkpoint(args.experiment_path, args.checkpoint, class_names, args.output_dir)
            if args.plot:
                # Generate plot for this specific checkpoint
                plot_single_checkpoint(args.experiment_path, args.checkpoint, class_names, args.output_dir)
        else:
            # Process all checkpoints in the experiment
            extract_all_checkpoints(args.experiment_path, class_names, args.output_dir)
            if args.plot:
                if args.plot_type == 'aggregate':
                    # Generate aggregate plot for all checkpoints
                    plot_aggregate_checkpoints(args.experiment_path, class_names, args.output_dir)
                else:
                    # Generate plots for each individual checkpoint
                    plot_all_individual_checkpoints(args.experiment_path, class_names, args.output_dir)
    else:
        # Process based on dataset and method filters
        process_experiments(args.log_dir, args.dataset, args.method, args.class_names, args.output_dir)
        
        if args.plot:
            print("Plotting functionality for batch processing not implemented yet.")
            print("Please use --experiment_path for specific experiment plotting.")

if __name__ == '__main__':
    main()
