import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def plot_class_distribution(json_file_path):
    """
    Plot class distribution across all checkpoints in the dataset.
    
    Args:
        json_file_path (str): Path to the train-all.json file
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to count occurrences of each class
    class_counts = defaultdict(int)
    
    # Iterate through all checkpoints
    for ckp_name, ckp_data in data.items():
        # Skip if not a checkpoint (in case there are other keys)
        if not isinstance(ckp_data, list):
            continue
            
        # Count each sample in the checkpoint
        for sample in ckp_data:
            # Check if sample has the required fields
            if isinstance(sample, dict) and 'common' in sample:
                common_name = sample['common']
                class_counts[common_name] += 1
    
    # Prepare data for plotting
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Sort by count for better visualization (descending order)
    sorted_data = sorted(zip(class_names, counts), key=lambda x: x[1], reverse=True)
    class_names, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(class_names)), counts, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                          '#DDA0DD', '#98D8C8', '#F39C12', '#E74C3C', '#9B59B6'])
    
    # Customize the plot
    plt.xlabel('Species (Common Name)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Occurrences', fontsize=12, fontweight='bold')
    plt.title('Train data Class Distribution Across All Checkpoints', fontsize=14, fontweight='bold')
    
    # Set x-axis labels with rotation for better readability
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Class Distribution Summary ===")
    print(f"Total unique species: {len(class_names)}")
    print(f"Total samples across all checkpoints: {sum(counts)}")
    print(f"\nSpecies counts (sorted by frequency):")
    for name, count in sorted_data:
        print(f"  {name}: {count} samples")
    
    return class_counts

def plot_class_distribution_by_checkpoint(json_file_path, prefix=''):
    """
    Plot class distribution broken down by checkpoint.
    
    Args:
        json_file_path (str): Path to the train-all.json file
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store counts by checkpoint
    ckp_class_counts = defaultdict(lambda: defaultdict(int))
    all_classes = set()
    
    # Collect data
    for ckp_name, ckp_data in data.items():
        if not isinstance(ckp_data, list):
            continue
            
        for sample in ckp_data:
            if isinstance(sample, dict) and 'common' in sample:
                common_name = sample['common']
                ckp_class_counts[ckp_name][common_name] += 1
                all_classes.add(common_name)
    
    # Prepare data for stacked bar chart
    checkpoints = sorted([ckp for ckp in ckp_class_counts.keys()])
    class_names = sorted(list(all_classes))
    
    # Create matrix for stacked bar chart
    data_matrix = []
    for class_name in class_names:
        row = [ckp_class_counts[ckp][class_name] for ckp in checkpoints]
        data_matrix.append(row)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Colors for different classes
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    bottom = np.zeros(len(checkpoints))
    
    for i, (class_name, row) in enumerate(zip(class_names, data_matrix)):
        ax.bar(checkpoints, row, bottom=bottom, label=class_name, color=colors[i])
        bottom += row
    
    ax.set_xlabel('Checkpoints', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution by Checkpoint', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{prefix}class_distribution_by_checkpoint.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Path to your JSON file
    json_file_path = "/fs/scratch/PAS2099/camera-trap-benchmark/orinoquia/orinoquia_N25/30/train-all.json"
    
    # Generate main class distribution plot
    print("Generating class distribution plot...")
    class_counts = plot_class_distribution(json_file_path)
    
    # Generate breakdown by checkpoint
    json_file_path = "/fs/scratch/PAS2099/camera-trap-benchmark/orinoquia/orinoquia_N25/30/train.json"
    print("\nGenerating class distribution by checkpoint plot...")
    plot_class_distribution_by_checkpoint(json_file_path, prefix='train_')
    json_file_path = "/fs/scratch/PAS2099/camera-trap-benchmark/orinoquia/orinoquia_N25/30/test.json"
    plot_class_distribution_by_checkpoint(json_file_path, prefix='test_')

    
    print("\nPlots saved as 'class_distribution.png' and 'class_distribution_by_checkpoint.png'")