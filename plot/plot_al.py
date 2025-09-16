import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_summary(file_path):
    """
    Load training summary from JSON file.
    
    Args:
        file_path (str): Path to the final_training_summary.json file
    
    Returns:
        dict: Checkpoint results data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('checkpoint_results', {})
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in - {file_path}")
        return {}

def plot_training_comparison():
    """
    Plot balanced accuracy comparison across different training settings.
    """
    
    # Define the training settings and their corresponding file paths
    training_settings = {
        'Zero-shot': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/zs/bioclip2/full_text_head/log/final_training_summary.json',
        # 'LoRA + CE + ALOE(10% data)': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/upper_bound_lora_ce_loss/bioclip2/lora_8_text_head/log/final_training_summary.json',
        'Full FT + CE + ALOE(10% data)': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/upper_bound_FT_ce_loss_frac0.1/bioclip2/full_text_head/aloe_percentage_0.1/log/final_training_summary.json',
        'Full FT + BSM+ ALOE(10% data)': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/upper_bound_bsm_loss_frac0.1/bioclip2/full_text_head/aloe_percentage_0.1/log/final_training_summary.json',
        'LoRA + BSM + ALOE(10% data)': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/aloe_percentage_0.1_num_samples_per_cluster_1/log/final_training_summary.json'
    }
    
    # Colors for each training setting
    colors = {
        'Zero-shot': '#FF6B6B',
        'LoRA + CE + ALOE(10% data)': "#4ECD98", 
        'Full FT + CE + ALOE(10% data)': '#96CEB4',
        'Full FT + BSM+ ALOE(10% data)': '#FFEAA7',
        'LoRA + BSM + ALOE(10% data)': '#DDA0DD'
    }
    
    # Line styles for variety
    line_styles = {
        'Zero-shot': '-',
        'LoRA + CE + ALOE(10% data)': '-.',
        'Full FT + CE + ALOE(10% data)': ':',
        'Full FT + BSM+ ALOE(10% data)': '-',
        'LoRA + BSM + ALOE(10% data)': '--'
    }
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Track all data for statistics
    all_results = {}
    
    # Plot each training setting
    for setting_name, file_path in training_settings.items():
        checkpoint_results = load_training_summary(file_path)
        
        if not checkpoint_results:
            print(f"Skipping {setting_name} - no data available")
            continue
            
        # Extract checkpoint numbers and balanced accuracy values
        checkpoints = sorted(checkpoint_results.keys(), key=lambda x: int(x.split('_')[1]))
        balanced_accuracies = [checkpoint_results[ckp]['balanced_accuracy'] for ckp in checkpoints]
        checkpoint_nums = [int(ckp.split('_')[1]) for ckp in checkpoints]
        
        # Store results for statistics
        all_results[setting_name] = {
            'checkpoints': checkpoint_nums,
            'accuracies': balanced_accuracies,
            'final_accuracy': balanced_accuracies[-1] if balanced_accuracies else 0
        }
        
        # Plot the curve
        plt.plot(checkpoint_nums, balanced_accuracies, 
                color=colors[setting_name],
                linestyle=line_styles[setting_name],
                marker='o', 
                markersize=6,
                linewidth=2.5,
                label=f'{setting_name})',
                alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Checkpoint', fontsize=12, fontweight='bold')
    plt.ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    plt.title('Balanced Accuracy Comparison\n(Orinoquia N25 Dataset)', 
              fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend with final accuracies
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set axis limits
    plt.xlim(0.5, 6.5)
    plt.ylim(0, 1.0)
    
    # Add horizontal line at 0.5 (random performance)
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/training_settings_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Training Settings Comparison Summary ===")
    print("Final Balanced Accuracy Rankings:")
    
    # Sort by final accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
    
    for i, (setting, results) in enumerate(sorted_results, 1):
        final_acc = results['final_accuracy']
        print(f"{i}. {setting}: {final_acc:.4f}")
    
    # Find best performing method
    if sorted_results:
        best_method, best_results = sorted_results[0]
        print(f"\n🏆 Best performing method: {best_method}")
        print(f"   Final accuracy: {best_results['final_accuracy']:.4f}")
        
        # Calculate improvement over zero-shot
        if 'Zero-shot' in all_results:
            zs_final = all_results['Zero-shot']['final_accuracy']
            improvement = ((best_results['final_accuracy'] - zs_final) / zs_final) * 100
            print(f"   Improvement over zero-shot: {improvement:+.1f}%")

def plot_sample_efficiency():
    """
    Plot showing relationship between number of samples and performance.
    """
    # This could be extended to show sample efficiency across checkpoints
    training_settings = {
        'Zero-shot': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/zs/bioclip2/full_text_head/log/final_training_summary.json',
        'LoRA + BSM': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/log/final_training_summary.json',
        'LoRA + BSM + ALOE': '/fs/scratch/PAS2099/camera-trap-final/dataAnalysis_logs/orinoquia_orinoquia_N25/upper_bound_lora_bsm_loss/bioclip2/lora_8_text_head/aloe_percentage_0.1_num_samples_per_cluster_1/log/final_training_summary.json'
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#DDA0DD']
    
    plt.figure(figsize=(10, 6))
    
    for i, (setting_name, file_path) in enumerate(training_settings.items()):
        checkpoint_results = load_training_summary(file_path)
        
        if not checkpoint_results:
            continue
            
        # Extract cumulative samples and balanced accuracy
        checkpoints = sorted(checkpoint_results.keys(), key=lambda x: int(x.split('_')[1]))
        cumulative_samples = []
        balanced_accuracies = []
        
        total_samples = 0
        for ckp in checkpoints:
            total_samples += checkpoint_results[ckp]['num_samples']
            cumulative_samples.append(total_samples)
            balanced_accuracies.append(checkpoint_results[ckp]['balanced_accuracy'])
        
        plt.plot(cumulative_samples, balanced_accuracies, 
                color=colors[i], marker='o', linewidth=2.5, markersize=6,
                label=setting_name)
    
    plt.xlabel('Cumulative Training Samples', fontsize=12, fontweight='bold')
    plt.ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    plt.title('Sample Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('plots/sample_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating training settings comparison plot...")
    plot_training_comparison()
    
    print("\nGenerating sample efficiency plot...")
    plot_sample_efficiency()
    
    print("\nPlots saved as 'training_settings_comparison.png' and 'sample_efficiency_comparison.png'")