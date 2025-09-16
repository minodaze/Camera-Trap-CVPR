import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def plot_false_predictions(json_file_path):
    """
    Plot incorrect predictions across all checkpoints.
    
    Args:
        json_file_path (str): Path to the final_image_level_predictions.json file
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to count false predictions
    false_pred_counts = defaultdict(int)
    
    # Iterate through all checkpoints
    for ckp_name, ckp_data in data.items():
        # Skip the 'stats' entry
        if ckp_name == 'stats':
            continue
            
        # Check if 'incorrect' key exists in checkpoint data
        if 'incorrect' in ckp_data:
            # Iterate through each true label in incorrect predictions
            for true_label, predictions in ckp_data['incorrect'].items():
                # Count each incorrect prediction
                for pred_info in predictions:
                    predicted_label = pred_info['prediction']
                    false_pred_counts[predicted_label] += 1
    
    # Prepare data for plotting
    prediction_labels = list(false_pred_counts.keys())
    counts = list(false_pred_counts.values())
    
    # Sort by count for better visualization
    sorted_data = sorted(zip(prediction_labels, counts), key=lambda x: x[1], reverse=True)
    prediction_labels, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(prediction_labels)), counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
    
    # Customize the plot
    plt.xlabel('Incorrect Prediction Labels', fontsize=12, fontweight='bold')
    plt.ylabel('Number of False Predictions', fontsize=12, fontweight='bold')
    plt.title('Distribution of Incorrect Predictions Across All Checkpoints', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(prediction_labels)), prediction_labels, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/false_predictions_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== False Predictions Summary ===")
    print(f"Total unique incorrect prediction labels: {len(prediction_labels)}")
    print(f"Total false predictions: {sum(counts)}")
    print(f"\nTop false predictions:")
    for label, count in sorted_data[:5]:
        print(f"  {label}: {count} times")

def plot_false_predictions_by_checkpoint(json_file_path):
    """
    Plot incorrect predictions broken down by checkpoint.
    
    Args:
        json_file_path (str): Path to the final_image_level_predictions.json file
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store counts by checkpoint
    ckp_false_preds = defaultdict(lambda: defaultdict(int))
    all_predictions = set()
    
    # Collect data
    for ckp_name, ckp_data in data.items():
        if ckp_name == 'stats':
            continue
            
        if 'incorrect' in ckp_data:
            for true_label, predictions in ckp_data['incorrect'].items():
                for pred_info in predictions:
                    predicted_label = pred_info['prediction']
                    ckp_false_preds[ckp_name][predicted_label] += 1
                    all_predictions.add(predicted_label)
    
    # Prepare data for stacked bar chart
    checkpoints = sorted([ckp for ckp in ckp_false_preds.keys()])
    prediction_labels = sorted(list(all_predictions))
    
    # Create matrix for stacked bar chart
    data_matrix = []
    for pred_label in prediction_labels:
        row = [ckp_false_preds[ckp][pred_label] for ckp in checkpoints]
        data_matrix.append(row)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors for different prediction labels
    colors = plt.cm.Set3(np.linspace(0, 1, len(prediction_labels)))
    
    bottom = np.zeros(len(checkpoints))
    bars = []
    
    for i, (pred_label, row) in enumerate(zip(prediction_labels, data_matrix)):
        bars.append(ax.bar(checkpoints, row, bottom=bottom, label=pred_label, color=colors[i]))
        bottom += row
    
    ax.set_xlabel('Checkpoints', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of False Predictions', fontsize=12, fontweight='bold')
    ax.set_title('False Predictions by Checkpoint', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/false_predictions_by_checkpoint.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Path to your JSON file
    json_file_path = "/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/upper_bound_ce_loss/bioclip2/full_text_head/log/final_image_level_predictions.json"
    
    # Generate both plots
    print("Generating false predictions distribution plot...")
    plot_false_predictions(json_file_path)
    
    print("\nGenerating false predictions by checkpoint plot...")
    plot_false_predictions_by_checkpoint(json_file_path)
    
    print("\nPlots saved as 'false_predictions_distribution.png' and 'false_predictions_by_checkpoint.png'")