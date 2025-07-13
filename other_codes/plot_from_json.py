#!/usr/bin/env python3
"""
Plot per-class accuracy from a JSON metrics file.

This script creates publication-style plots similar to the example figure,
showing per-class accuracy with class index on x-axis and accuracy on y-axis.

Usage:
    python plot_from_json.py --json_file path/to/metrics.json [--output output.png]
    
Example:
    python plot_from_json.py --json_file extracted_metrics/bioclip2_2025-07-06-22-13-54_scientific_name_ckp_1_per_class_metrics.json --output figures/per_class_accuracy.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_metrics_from_json(json_file):
    """Load metrics from JSON file."""
    try:
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return None

def plot_per_class_accuracy_from_json(metrics, output_path=None, style='publication'):
    """
    Plot per-class accuracy from metrics dictionary.
    
    Args:
        metrics: Dictionary containing per-class metrics
        output_path: Path to save the plot
        style: 'publication' for clean style, 'detailed' for more info
    """
    per_class_acc = metrics['per_class_accuracy']
    class_names = metrics.get('class_names', [])
    samples_per_class = metrics.get('samples_per_class', [])
    
    n_classes = len(per_class_acc)
    class_indices = np.arange(1, n_classes + 1)  # Start from 1 like in the example
    
    # Convert accuracy to percentage
    per_class_acc_pct = [acc * 100 for acc in per_class_acc]
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if style == 'publication':
        # Publication-style plot similar to the example
        # Plot line with markers
        line = ax.plot(class_indices, per_class_acc_pct, 
                      marker='o', linewidth=2, markersize=6,
                      color='blue', label='Test Accuracy')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set labels and title
        ax.set_xlabel('Class Index', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        
        # Set axis limits
        ax.set_xlim(0.5, n_classes + 0.5)
        ax.set_ylim(0, 105)
        
        # Set x-axis ticks
        ax.set_xticks(class_indices)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add overall metrics as text
        overall_acc = metrics.get('overall_accuracy', 0) * 100
        balanced_acc = metrics.get('balanced_accuracy', 0) * 100
        
        textstr = f'Overall Accuracy: {overall_acc:.1f}%\nBalanced Accuracy: {balanced_acc:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    elif style == 'detailed':
        # More detailed plot with class names and sample counts
        # Create bar plot with different colors based on accuracy
        colors = ['red' if acc < 0.5 else 'orange' if acc < 0.8 else 'green' 
                 for acc in per_class_acc]
        
        bars = ax.bar(class_indices, per_class_acc_pct, color=colors, alpha=0.7)
        
        # Add accuracy values on top of bars
        for i, (bar, acc) in enumerate(zip(bars, per_class_acc_pct)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Set labels
        ax.set_xlabel('Class Index', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        
        # Set x-axis ticks and labels
        ax.set_xticks(class_indices)
        if class_names:
            # Show class names if available (rotate for readability)
            ax.set_xticklabels([f'{i}\n{name[:15]}...' if len(name) > 15 else f'{i}\n{name}' 
                               for i, name in enumerate(class_indices, 1)], 
                              rotation=45, ha='right', fontsize=8)
        
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add experiment info as subtitle if available
    checkpoint = metrics.get('checkpoint', '')
    method = metrics.get('method', '')
    dataset = metrics.get('dataset', '')
    
    if checkpoint or method or dataset:
        subtitle_parts = []
        if dataset:
            subtitle_parts.append(f"Dataset: {dataset}")
        if method:
            subtitle_parts.append(f"Method: {method}")
        if checkpoint:
            subtitle_parts.append(f"Checkpoint: {checkpoint}")
        
        subtitle = " | ".join(subtitle_parts)
        plt.suptitle(subtitle, fontsize=10, y=0.95)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nPer-class Accuracy Summary:")
    print(f"Overall Accuracy: {metrics.get('overall_accuracy', 0):.4f}")
    print(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"Number of classes: {n_classes}")
    
    if class_names:
        print("\nDetailed per-class results:")
        for i, (name, acc, samples) in enumerate(zip(class_names, per_class_acc, samples_per_class)):
            print(f"  Class {i+1:2d} ({name[:30]:30s}): {acc:.3f} ({samples:3d} samples)")
    
    # Save plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()

def plot_multiple_checkpoints_comparison(json_files, output_path=None):
    """
    Plot comparison of multiple checkpoints from different JSON files.
    
    Args:
        json_files: List of JSON file paths
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    all_metrics = []
    
    for i, json_file in enumerate(json_files):
        metrics = load_metrics_from_json(json_file)
        if metrics is None:
            continue
            
        all_metrics.append(metrics)
        
        per_class_acc = metrics['per_class_accuracy']
        n_classes = len(per_class_acc)
        class_indices = np.arange(1, n_classes + 1)
        per_class_acc_pct = [acc * 100 for acc in per_class_acc]
        
        checkpoint = metrics.get('checkpoint', f'File {i+1}')
        method = metrics.get('method', '')
        
        label = f"{checkpoint}"
        if method:
            label += f" ({method})"
        
        ax.plot(class_indices, per_class_acc_pct, 
               marker=markers[i % len(markers)], 
               color=colors[i % len(colors)],
               linewidth=2, markersize=6, label=label)
    
    if all_metrics:
        # Use the first file's class configuration for x-axis
        n_classes = len(all_metrics[0]['per_class_accuracy'])
        class_indices = np.arange(1, n_classes + 1)
        
        ax.set_xlabel('Class Index', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0.5, n_classes + 0.5)
        ax.set_ylim(0, 105)
        ax.set_xticks(class_indices)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {output_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot per-class accuracy from JSON metrics file')
    parser.add_argument('--json_file', type=str, required=True,
                       help='Path to JSON metrics file')
    parser.add_argument('--output', type=str,
                       help='Output path for the plot (optional)')
    parser.add_argument('--style', type=str, choices=['publication', 'detailed'], 
                       default='publication',
                       help='Plot style: publication (clean) or detailed (with class names)')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Compare multiple JSON files (provide multiple file paths)')
    
    args = parser.parse_args()
    
    if args.compare:
        # Plot comparison of multiple files
        json_files = args.compare
        if args.json_file not in json_files:
            json_files = [args.json_file] + json_files
        
        plot_multiple_checkpoints_comparison(json_files, args.output)
    else:
        # Plot single file
        if not os.path.exists(args.json_file):
            print(f"Error: JSON file does not exist: {args.json_file}")
            return
        
        metrics = load_metrics_from_json(args.json_file)
        if metrics is None:
            return
        
        plot_per_class_accuracy_from_json(metrics, args.output, args.style)

if __name__ == '__main__':
    main()
