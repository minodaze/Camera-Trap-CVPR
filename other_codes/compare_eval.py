import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse log file to extract learning rate and balanced accuracy per checkpoint"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract learning rate
    lr_match = re.search(r"'lr': ([\d.e-]+)", content)
    lr = float(lr_match.group(1)) if lr_match else None
    
    # Extract balanced accuracy for each checkpoint
    ckpt_pattern = r"Training on checkpoint ckp_(\d+)\..*?balanced acc: ([\d.]+)"
    matches = re.findall(ckpt_pattern, content, re.DOTALL)
    
    checkpoints = []
    balanced_accs = []
    
    for ckpt_num, balanced_acc in matches:
        checkpoints.append(int(ckpt_num))
        balanced_accs.append(float(balanced_acc))
    
    return lr, checkpoints, balanced_accs

def plot_multiple_logs(log_files, output_path=None):
    """Plot balanced accuracy vs checkpoint for multiple log files and print statistics"""
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(log_files)))
    
    print("\n" + "="*60)
    print("BALANCED ACCURACY ANALYSIS")
    print("="*60)
    
    all_stats = []
    
    for i, log_file in enumerate(log_files):
        lr, checkpoints, balanced_accs = parse_log_file(log_file)
        
        if lr is not None and checkpoints:
            label = f'lr={lr:.1e}'
            plt.plot(checkpoints, balanced_accs, 'o-', color=colors[i], 
                    label=label, linewidth=2, markersize=6)
            
            # Calculate statistics
            avg_balanced_acc = np.mean(balanced_accs)
            std_balanced_acc = np.std(balanced_accs)
            min_balanced_acc = np.min(balanced_accs)
            max_balanced_acc = np.max(balanced_accs)
            final_balanced_acc = balanced_accs[-1] if balanced_accs else 0
            
            # Store stats for comparison
            all_stats.append({
                'log_file': log_file,
                'lr': lr,
                'avg': avg_balanced_acc,
                'std': std_balanced_acc,
                'min': min_balanced_acc,
                'max': max_balanced_acc,
                'final': final_balanced_acc,
                'checkpoints': len(checkpoints)
            })
            
            # Print individual statistics
            print(f"\nRun {i+1}: {label}")
            print(f"  Log file: {log_file.split('/')[-3:]}")  # Show last 3 path components
            print(f"  Checkpoints: {len(checkpoints)}")
            print(f"  Average Balanced Acc: {avg_balanced_acc:.4f}")
            print(f"  Std Dev: {std_balanced_acc:.4f}")
            print(f"  Min: {min_balanced_acc:.4f}")
            print(f"  Max: {max_balanced_acc:.4f}")
            print(f"  Final: {final_balanced_acc:.4f}")
    
    # Print comparison summary
    if len(all_stats) > 1:
        print(f"\n" + "-"*60)
        print("COMPARISON SUMMARY")
        print("-"*60)
        
        # Sort by average balanced accuracy (descending)
        sorted_stats = sorted(all_stats, key=lambda x: x['avg'], reverse=True)
        
        print(f"{'Rank':<5} {'LR':<12} {'Avg Bal Acc':<12} {'Final Bal Acc':<14} {'Checkpoints':<12}")
        print("-" * 60)
        
        for rank, stats in enumerate(sorted_stats, 1):
            print(f"{rank:<5} {stats['lr']:<12.1e} {stats['avg']:<12.4f} {stats['final']:<14.4f} {stats['checkpoints']:<12}")
        
        # Calculate overall statistics
        all_avgs = [s['avg'] for s in all_stats]
        overall_mean = np.mean(all_avgs)
        overall_std = np.std(all_avgs)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Mean of averages: {overall_mean:.4f}")
        print(f"  Std of averages: {overall_std:.4f}")
        print(f"  Range: {np.min(all_avgs):.4f} - {np.max(all_avgs):.4f}")
        
        # Check for reproducibility (if multiple runs with same LR)
        lr_groups = {}
        for stats in all_stats:
            lr = stats['lr']
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(stats['avg'])
        
        print(f"\nREPRODUCIBILITY CHECK:")
        for lr, avgs in lr_groups.items():
            if len(avgs) > 1:
                lr_std = np.std(avgs)
                print(f"  LR {lr:.1e}: {len(avgs)} runs, avg std = {lr_std:.6f}")
                if lr_std < 0.001:
                    print(f"    ✓ Highly reproducible (std < 0.001)")
                elif lr_std < 0.01:
                    print(f"    ~ Moderately reproducible (std < 0.01)")
                else:
                    print(f"    ✗ Low reproducibility (std >= 0.01)")
    
    plt.xlabel('Checkpoint')
    plt.ylabel('Balanced Accuracy')
    plt.title('Balanced Accuracy vs Checkpoint')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
    
    print("="*60)
    plt.show()
    
    return all_stats

def analyze_logs_stats_only(log_files):
    """Analyze log files and print statistics without plotting"""
    print("\n" + "="*60)
    print("BALANCED ACCURACY ANALYSIS (Stats Only)")
    print("="*60)
    
    all_stats = []
    
    for i, log_file in enumerate(log_files):
        lr, checkpoints, balanced_accs = parse_log_file(log_file)
        
        if lr is not None and checkpoints:
            # Calculate statistics
            avg_balanced_acc = np.mean(balanced_accs)
            std_balanced_acc = np.std(balanced_accs)
            min_balanced_acc = np.min(balanced_accs)
            max_balanced_acc = np.max(balanced_accs)
            final_balanced_acc = balanced_accs[-1] if balanced_accs else 0
            
            # Store stats for comparison
            all_stats.append({
                'log_file': log_file,
                'lr': lr,
                'avg': avg_balanced_acc,
                'std': std_balanced_acc,
                'min': min_balanced_acc,
                'max': max_balanced_acc,
                'final': final_balanced_acc,
                'checkpoints': len(checkpoints)
            })
            
            # Print individual statistics
            print(f"\nRun {i+1}: lr={lr:.1e}")
            print(f"  Log file: .../{'/'.join(log_file.split('/')[-3:])}")  # Show last 3 path components
            print(f"  Checkpoints: {len(checkpoints)}")
            print(f"  Average Balanced Acc: {avg_balanced_acc:.4f}")
            print(f"  Std Dev: {std_balanced_acc:.4f}")
            print(f"  Min: {min_balanced_acc:.4f}")
            print(f"  Max: {max_balanced_acc:.4f}")
            print(f"  Final: {final_balanced_acc:.4f}")
    
    # Print comparison summary if multiple logs
    if len(all_stats) > 1:
        print(f"\n" + "-"*60)
        print("COMPARISON SUMMARY")
        print("-"*60)
        
        # Sort by average balanced accuracy (descending)
        sorted_stats = sorted(all_stats, key=lambda x: x['avg'], reverse=True)
        
        print(f"{'Rank':<5} {'LR':<12} {'Avg Bal Acc':<12} {'Final Bal Acc':<14} {'Checkpoints':<12}")
        print("-" * 60)
        
        for rank, stats in enumerate(sorted_stats, 1):
            print(f"{rank:<5} {stats['lr']:<12.1e} {stats['avg']:<12.4f} {stats['final']:<14.4f} {stats['checkpoints']:<12}")
        
        # Calculate overall statistics
        all_avgs = [s['avg'] for s in all_stats]
        overall_mean = np.mean(all_avgs)
        overall_std = np.std(all_avgs)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Mean of averages: {overall_mean:.4f}")
        print(f"  Std of averages: {overall_std:.4f}")
        print(f"  Range: {np.min(all_avgs):.4f} - {np.max(all_avgs):.4f}")
        
        # Check for reproducibility (if multiple runs with same LR)
        lr_groups = {}
        for stats in all_stats:
            lr = stats['lr']
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(stats['avg'])
        
        print(f"\nREPRODUCIBILITY CHECK:")
        for lr, avgs in lr_groups.items():
            if len(avgs) > 1:
                lr_std = np.std(avgs)
                print(f"  LR {lr:.1e}: {len(avgs)} runs, avg std = {lr_std:.6f}")
                if lr_std < 0.001:
                    print(f"    ✓ Highly reproducible (std < 0.001)")
                elif lr_std < 0.01:
                    print(f"    ~ Moderately reproducible (std < 0.01)")
                else:
                    print(f"    ✗ Low reproducibility (std >= 0.01)")
    
    print("="*60)
    return all_stats

# Example usage for single log file
if __name__ == "__main__":
    # For single log file
    # log_file = "log.txt"  # Replace with your log file path
    
    # if Path(log_file).exists():
    #     lr, checkpoints, balanced_accs = parse_log_file(log_file)
        
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(checkpoints, balanced_accs, 'o-', linewidth=2, markersize=6, 
    #             label=f'lr={lr:.1e}' if lr else 'Unknown LR')
    #     plt.xlabel('Checkpoint')
    #     plt.ylabel('Balanced Accuracy')
    #     plt.title('Balanced Accuracy vs Checkpoint')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print(f"Log file {log_file} not found")
    
    # For multiple log files, use:
    log_files = [
        "/fs/scratch/PAS2099/sooyoung/icicle/log_auto/pipeline/swg_swg_loc_0182/upper_bound/lr_0.000025/2025-07-12-20-0c-5e/bioclip2/full_text_head/2025-07-12-20-6021-00/log/log.txt", 
        "/fs/scratch/PAS2099/sooyoung/icicle/log_auto/pipeline/swg_swg_loc_0182/upper_bound/lr_0.000025/2025-07-12-20-0c-5e/bioclip2/full_text_head/2025-07-12-20-6776-00/log/log.txt", 
    ]
    
    # Option 1: Full analysis with plot and statistics
    stats = plot_multiple_logs(log_files, "balanced_acc_comparison.png")
    print("\nStatistics for all runs:", stats)
    # Option 2: Statistics only (uncomment if you just want stats without plot)
    # stats = analyze_logs_stats_only(log_files)
    # analyze_logs_stats_only(log_files)  # Uncomment to run stats only analysis