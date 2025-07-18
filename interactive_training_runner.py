#!/usr/bin/env python3
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_training(gpu_id, dataset, setting, config_root, workspace_root, conda_env):
    """Run training on a specific GPU with the given configuration"""
    
    # Set CUDA_VISIBLE_DEVICES for this process
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Convert dataset name from format like "MAD/MAD_MAD01" to "MAD_MAD_MAD01"
    dataset_converted = dataset.replace('/', '_')
    
    # Determine the config file and additional arguments
    # Get YAML file name by splitting dataset by '_', removing first part, keeping rest
    dataset_parts = dataset_converted.split('_')
    if len(dataset_parts) > 1:
        yaml_name = '_'.join(dataset_parts[1:])
    else:
        yaml_name = dataset_converted
    
    if setting in ['lora_ce', 'full_ce']:
        config_file = f"{config_root}/{dataset_converted}/{yaml_name}_accu_ce.yaml"
    else:  # lora_bsm, full_bsm
        config_file = f"{config_root}/{dataset_converted}/{yaml_name}_accu_bsm.yaml"
    
    # Build the command with default arguments
    cmd = [
        'conda', 'run', '-n', conda_env,
        'python', 'run_pipeline.py',
        '--wandb',
        '--eval_per_epoch', 
        '--test_per_epoch',
        '--save_best_model',
        '--c', config_file
    ]
    
    if setting.startswith('lora'):
        cmd.extend(['--lora_bottleneck', '8'])
    else:  # full training
        cmd.append('--full')
    
    print(f"🚀 Starting GPU {gpu_id}: {dataset_converted} with {setting}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {workspace_root}")
    print("-" * 50)
    
    # Run the training
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=None
        )
        
        if result.returncode == 0:
            print(f"✅ GPU {gpu_id} ({dataset_converted}, {setting}) completed successfully")
            return True, gpu_id, dataset_converted, setting, result.stdout, result.stderr
        else:
            print(f"❌ GPU {gpu_id} ({dataset_converted}, {setting}) failed with return code {result.returncode}")
            return False, gpu_id, dataset_converted, setting, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ GPU {gpu_id} ({dataset_converted}, {setting}) timed out")
        return False, gpu_id, dataset_converted, setting, "", "Training timed out"
    except Exception as e:
        print(f"💥 GPU {gpu_id} ({dataset_converted}, {setting}) failed with exception: {str(e)}")
        return False, gpu_id, dataset_converted, setting, "", str(e)

def main():
    if len(sys.argv) != 4:
        print("Usage: python interactive_training_runner.py <training_configs> <workspace_root> <conda_env>")
        sys.exit(1)
    
    training_configs = sys.argv[1]
    workspace_root = sys.argv[2]
    conda_env = sys.argv[3]
    config_root = "/fs/scratch/PAS2099/camera-trap-final/configs"
    
    # Parse training configurations
    training_tasks = []
    for config in training_configs.split(','):
        gpu_id, dataset, setting = config.split(':')
        training_tasks.append((int(gpu_id), dataset, setting))
    
    print(f"📋 Training Tasks:")
    for gpu_id, dataset, setting in training_tasks:
        dataset_converted = dataset.replace('/', '_')
        print(f"  GPU {gpu_id}: {dataset_converted} → {setting}")
    print("")
    
    # Run training tasks in parallel
    with ThreadPoolExecutor(max_workers=len(training_tasks)) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_training, gpu_id, dataset, setting, config_root, workspace_root, conda_env): (gpu_id, dataset, setting)
            for gpu_id, dataset, setting in training_tasks
        }
        
        # Wait for completion and collect results
        results = []
        for future in as_completed(future_to_task):
            gpu_id, dataset, setting = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"💥 Exception in GPU {gpu_id} ({dataset}, {setting}): {str(e)}")
                results.append((False, gpu_id, dataset, setting, "", str(e)))
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🏁 TRAINING SUMMARY")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for success, gpu_id, dataset, setting, stdout, stderr in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: GPU {gpu_id} - {dataset} ({setting})")
        if not success and stderr:
            print(f"  Error: {stderr[:200]}...")
        successful += 1 if success else 0
        failed += 1 if not success else 0
    
    print(f"\n📊 Results: {successful} successful, {failed} failed")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
