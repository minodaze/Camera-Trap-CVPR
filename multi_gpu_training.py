#!/usr/bin/env python3
"""
Multi-GPU Training Automation Script for ICICLE Benchmark

This script manages training across 4 GPUs, with each GPU handling one dataset
and cycling through 4 different training settings.

Usage:
    python multi_gpu_training.py
    (Interactive mode - will prompt for datasets)
    
    python multi_gpu_training.py --datasets dataset1 dataset2 dataset3 dataset4
    (Command line mode)
"""

import os
import sys
import time
import subprocess
import threading
import argparse
import logging
import json
from pathlib import Path
from queue import Queue
import signal
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_gpu_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'

def print_banner():
    """Print a nice banner"""
    print(f"\n{Colors.BRIGHT_CYAN}=" * 70)
    print(f"🚀 ICICLE Benchmark Multi-GPU Training Automation")
    print(f"=" * 70 + Colors.RESET)
    print(f"{Colors.YELLOW}Training Configuration:{Colors.RESET}")
    print(f"  • 4 GPUs running in parallel")
    print(f"  • 4 training settings per GPU")
    print(f"  • Sequential execution within each GPU")
    print(f"  • Automatic GPU memory management")
    print(f"{Colors.BRIGHT_CYAN}=" * 70 + Colors.RESET + "\n")

def get_available_datasets(config_root):
    """Get list of available datasets"""
    try:
        datasets = []
        config_path = Path(config_root)
        if config_path.exists():
            for item in config_path.iterdir():
                if item.is_dir():
                    datasets.append(item.name)
        return sorted(datasets)
    except Exception as e:
        logger.error(f"Error reading datasets from {config_root}: {e}")
        return []

def select_datasets_interactive(config_root):
    """Interactive dataset selection"""
    print(f"{Colors.BRIGHT_YELLOW}📂 Dataset Selection{Colors.RESET}")
    print("=" * 50)
    
    # Get available datasets
    available_datasets = get_available_datasets(config_root)
    
    if not available_datasets:
        print(f"{Colors.RED}❌ No datasets found in {config_root}{Colors.RESET}")
        return None
    
    print(f"{Colors.CYAN}Available datasets:{Colors.RESET}")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"  {i:2d}. {dataset}")
    
    print(f"\n{Colors.YELLOW}Please enter 4 dataset names for the 4 GPUs:{Colors.RESET}")
    
    selected_datasets = []
    for gpu_id in range(4):
        while True:
            dataset = input(f"{Colors.GREEN}GPU {gpu_id}:{Colors.RESET} ").strip()
            
            if dataset in available_datasets:
                selected_datasets.append(dataset)
                print(f"  ✅ GPU {gpu_id} will use dataset: {Colors.BRIGHT_GREEN}{dataset}{Colors.RESET}")
                break
            else:
                print(f"  {Colors.RED}❌ Dataset '{dataset}' not found. Please choose from the list above.{Colors.RESET}")
    
    return selected_datasets

class GPUTrainer:
    """Handles training execution on a specific GPU"""
    
    def __init__(self, gpu_id, dataset, config_root_path, workspace_path, conda_env="icicle"):
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.config_root_path = Path(config_root_path)
        self.workspace_path = Path(workspace_path)
        self.conda_env = conda_env
        self.current_process = None
        self.training_queue = Queue()
        self.is_running = False
        self.completed_trainings = []
        self.failed_trainings = []
        
        # Get shortened dataset name for YAML files
        # Example: MAD_MAD_A05 -> MAD_A05 (split by '_', use from 2nd chunk onwards)
        dataset_parts = dataset.split('_')
        if len(dataset_parts) > 1:
            self.yaml_dataset_name = '_'.join(dataset_parts[1:])
        else:
            self.yaml_dataset_name = dataset
        
        # Define training settings in the specified order
        self.training_settings = [
            {
                'config': f'{self.yaml_dataset_name}_accu_bsm.yaml',
                'args': ['--lora_bottleneck', '8'],
                'name': 'LoRA BSM (bottleneck=8)',
                'short_name': 'lora_bsm_8'
            },
            {
                'config': f'{self.yaml_dataset_name}_accu_ce.yaml', 
                'args': ['--lora_bottleneck', '8'],
                'name': 'LoRA CE (bottleneck=8)',
                'short_name': 'lora_ce_8'
            },
            {
                'config': f'{self.yaml_dataset_name}_accu_ce.yaml',
                'args': ['--full'],
                'name': 'Full CE',
                'short_name': 'full_ce'
            },
            {
                'config': f'{self.yaml_dataset_name}_accu_bsm.yaml',
                'args': ['--full'], 
                'name': 'Full BSM',
                'short_name': 'full_bsm'
            }
        ]
        
        # Populate training queue
        for i, setting in enumerate(self.training_settings):
            self.training_queue.put((i + 1, setting))
    
    def log(self, message, color=None):
        """Log message with GPU-specific prefix and optional color"""
        prefix = f"[GPU-{self.gpu_id}][{self.dataset}]"
        if color:
            message = f"{color}{message}{Colors.RESET}"
        logger.info(f"{prefix} {message}")
    
    def log_yaml_info(self):
        """Log information about dataset name conversion"""
        self.log(f"📁 Dataset directory: {self.dataset}", Colors.CYAN)
        self.log(f"📄 YAML prefix: {self.yaml_dataset_name}", Colors.CYAN)
    
    def clear_gpu_memory(self):
        """Clear GPU memory before starting new training"""
        self.log("🧹 Clearing GPU memory...", Colors.YELLOW)
        try:
            # Use torch to clear GPU memory with conda environment
            clear_cmd = [
                'conda', 'run', '-n', self.conda_env,
                'python', '-c', 
                f"""
import torch
import gc
try:
    if torch.cuda.is_available():
        device = torch.device('cuda:{self.gpu_id}')
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print(f'GPU {self.gpu_id} memory cleared successfully')
    else:
        print('CUDA not available')
except Exception as e:
    print(f'Error clearing memory: {{e}}')
"""
            ]
            
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            
            result = subprocess.run(clear_cmd, env=env, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.log("✅ GPU memory cleared successfully", Colors.GREEN)
            else:
                self.log(f"⚠️ GPU memory clear returned code {result.returncode}", Colors.YELLOW)
                if result.stderr:
                    self.log(f"Error: {result.stderr.strip()}", Colors.YELLOW)
                    
        except subprocess.TimeoutExpired:
            self.log("⏰ GPU memory clear timed out", Colors.YELLOW)
        except Exception as e:
            self.log(f"❌ Error clearing GPU memory: {e}", Colors.YELLOW)
        
        # Additional safety wait
        time.sleep(3)
    
    def validate_config_file(self, setting):
        """Validate that config file exists"""
        config_path = self.config_root_path / self.dataset / setting['config']
        return config_path.exists(), config_path
    
    def build_command(self, setting):
        """Build the training command for a specific setting"""
        config_exists, config_path = self.validate_config_file(setting)
        
        if not config_exists:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Base command using conda environment
        cmd = [
            'conda', 'run', '-n', self.conda_env,
            'python', 
            str(self.workspace_path / 'run_pipeline.py'),
            '--wandb',
            '--eval_per_epoch', 
            '--test_per_epoch',
            '--save_best_model',
            '--c', str(config_path)
        ]
        
        # Add specific arguments
        cmd.extend(setting['args'])
        
        return cmd
    
    def run_training(self, training_num, setting):
        """Run a single training session"""
        self.log(f"🚀 Starting training {training_num}/4: {setting['name']}", Colors.BRIGHT_CYAN)
        
        try:
            # Clear GPU memory before training
            self.clear_gpu_memory()
            
            # Build command
            cmd = self.build_command(setting)
            self.log(f"💻 Command: {' '.join(cmd)}", Colors.BLUE)
            
            # Set environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            
            # Create log files for this training
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = self.workspace_path / 'logs' / f'gpu_{self.gpu_id}_{self.dataset}'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            stdout_log = log_dir / f'training_{training_num}_{setting["short_name"]}_{timestamp}.out'
            stderr_log = log_dir / f'training_{training_num}_{setting["short_name"]}_{timestamp}.err'
            
            # Start training process
            with open(stdout_log, 'w') as stdout_f, open(stderr_log, 'w') as stderr_f:
                self.current_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    cwd=str(self.workspace_path)
                )
                
                self.log(f"🔥 Training started with PID {self.current_process.pid}", Colors.GREEN)
                self.log(f"📄 Logs: {stdout_log.name} | {stderr_log.name}", Colors.CYAN)
                
                # Wait for completion
                return_code = self.current_process.wait()
                
                if return_code == 0:
                    self.log(f"🎉 Training {training_num} completed successfully!", Colors.BRIGHT_GREEN)
                    self.completed_trainings.append((training_num, setting['name']))
                    return True
                else:
                    self.log(f"💥 Training {training_num} failed with return code {return_code}", Colors.BRIGHT_RED)
                    self.failed_trainings.append((training_num, setting['name'], return_code))
                    
                    # Log last few lines of error log
                    if stderr_log.exists():
                        try:
                            with open(stderr_log, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    self.log("Last error lines:", Colors.RED)
                                    for line in lines[-3:]:
                                        self.log(f"  {line.strip()}", Colors.RED)
                        except:
                            pass
                    return False
                    
        except FileNotFoundError as e:
            self.log(f"💥 Training {training_num} failed: {e}", Colors.BRIGHT_RED)
            self.failed_trainings.append((training_num, setting['name'], str(e)))
            return False
        except Exception as e:
            self.log(f"💥 Training {training_num} failed with error: {e}", Colors.BRIGHT_RED)
            self.failed_trainings.append((training_num, setting['name'], str(e)))
            return False
        finally:
            self.current_process = None
    
    def run_all_trainings(self):
        """Run all training settings for this GPU/dataset"""
        self.is_running = True
        self.log(f"🎯 Starting training sequence for dataset {self.dataset}", Colors.BRIGHT_MAGENTA)
        self.log_yaml_info()
        
        try:
            while not self.training_queue.empty() and self.is_running:
                training_num, setting = self.training_queue.get()
                
                success = self.run_training(training_num, setting)
                
                if not success:
                    self.log(f"⚠️ Training {training_num} failed, but continuing with next training", Colors.YELLOW)
                
                # Brief pause between trainings
                if not self.training_queue.empty():
                    self.log("⏳ Waiting 10 seconds before next training...", Colors.CYAN)
                    time.sleep(10)
                    
        except KeyboardInterrupt:
            self.log("⛔ Training interrupted by user", Colors.YELLOW)
        finally:
            self.is_running = False
            self.log("✅ Training sequence completed", Colors.BRIGHT_GREEN)
            self.print_summary()
    
    def stop(self):
        """Stop current training and clear queue"""
        self.is_running = False
        if self.current_process:
            self.log("🛑 Stopping current training process...", Colors.YELLOW)
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.log("💀 Force killing training process...", Colors.RED)
                self.current_process.kill()
            except Exception as e:
                self.log(f"❌ Error stopping process: {e}", Colors.RED)
    
    def print_summary(self):
        """Print training summary for this GPU"""
        self.log("=" * 50, Colors.BRIGHT_CYAN)
        self.log("📊 TRAINING SUMMARY", Colors.BRIGHT_CYAN)
        self.log("=" * 50, Colors.BRIGHT_CYAN)
        
        self.log(f"✅ Completed trainings: {len(self.completed_trainings)}", Colors.GREEN)
        for training_num, name in self.completed_trainings:
            self.log(f"  ✓ Training {training_num}: {name}", Colors.GREEN)
        
        if self.failed_trainings:
            self.log(f"❌ Failed trainings: {len(self.failed_trainings)}", Colors.RED)
            for training_num, name, error in self.failed_trainings:
                self.log(f"  ✗ Training {training_num}: {name} - {error}", Colors.RED)
        
        self.log("=" * 50, Colors.BRIGHT_CYAN)

class MultiGPUTrainingManager:
    """Manages training across multiple GPUs"""
    
    def __init__(self, datasets, config_root_path, workspace_path, conda_env="icicle"):
        self.datasets = datasets
        self.config_root_path = config_root_path
        self.workspace_path = workspace_path
        self.conda_env = conda_env
        self.trainers = []
        self.threads = []
        self.is_running = False
        
        # Validate inputs
        if len(datasets) != 4:
            raise ValueError("Exactly 4 datasets must be provided for 4 GPUs")
        
        # Create trainers for each GPU
        for gpu_id, dataset in enumerate(datasets):
            trainer = GPUTrainer(gpu_id, dataset, config_root_path, workspace_path, conda_env)
            self.trainers.append(trainer)
    
    def validate_configs(self):
        """Validate that all required config files exist"""
        logger.info("🔍 Validating configuration files...")
        
        missing_configs = []
        for trainer in self.trainers:
            for setting in trainer.training_settings:
                config_exists, config_path = trainer.validate_config_file(setting)
                if not config_exists:
                    missing_configs.append(str(config_path))
        
        if missing_configs:
            logger.error("❌ Missing configuration files:")
            for config in missing_configs:
                logger.error(f"  - {config}")
            raise FileNotFoundError("Some configuration files are missing")
        
        logger.info("✅ All configuration files found")
    
    def print_training_plan(self):
        """Print the training plan"""
        print(f"\n{Colors.BRIGHT_YELLOW}📋 Training Plan{Colors.RESET}")
        print("=" * 60)
        
        for trainer in self.trainers:
            print(f"\n{Colors.CYAN}GPU {trainer.gpu_id} - Dataset: {Colors.BRIGHT_GREEN}{trainer.dataset}{Colors.RESET}")
            print(f"  📄 YAML files use prefix: {Colors.YELLOW}{trainer.yaml_dataset_name}{Colors.RESET}")
            for i, setting in enumerate(trainer.training_settings, 1):
                print(f"  {i}. {setting['name']}")
                print(f"     Config: {setting['config']}")
                print(f"     Args: {' '.join(setting['args'])}")
        
        print(f"\n{Colors.BRIGHT_CYAN}Total: 16 trainings (4 GPUs × 4 settings each){Colors.RESET}")
        print("=" * 60)
    
    def start_training(self):
        """Start training on all GPUs"""
        print(f"\n{Colors.BRIGHT_CYAN}🚀 Starting Multi-GPU Training{Colors.RESET}")
        print("=" * 60)
        print(f"📂 Config root: {self.config_root_path}")
        print(f"💼 Workspace: {self.workspace_path}")
        print(f"� Conda environment: {Colors.BRIGHT_GREEN}{self.conda_env}{Colors.RESET}")
        print(f"�🗂️  Datasets: {', '.join(self.datasets)}")
        print("=" * 60)
        
        try:
            # Validate configurations
            self.validate_configs()
            
            # Show training plan
            self.print_training_plan()
            
            # Ask for confirmation
            print(f"\n{Colors.YELLOW}Do you want to start the training? (y/N): {Colors.RESET}", end="")
            response = input().strip().lower()
            
            if response not in ['y', 'yes']:
                print(f"{Colors.YELLOW}Training cancelled by user.{Colors.RESET}")
                return
            
            self.is_running = True
            
            # Start training threads for each GPU
            print(f"\n{Colors.BRIGHT_GREEN}🔥 Starting training on all GPUs...{Colors.RESET}")
            for trainer in self.trainers:
                thread = threading.Thread(
                    target=trainer.run_all_trainings,
                    name=f"GPU-{trainer.gpu_id}-{trainer.dataset}"
                )
                thread.daemon = True
                self.threads.append(thread)
                thread.start()
                logger.info(f"Started training thread for GPU {trainer.gpu_id} with dataset {trainer.dataset}")
            
            # Wait for all threads to complete
            for thread in self.threads:
                thread.join()
                
            logger.info(f"{Colors.BRIGHT_GREEN}🎉 All training completed!{Colors.RESET}")
            
        except KeyboardInterrupt:
            logger.info("⛔ Training interrupted by user")
            self.stop_all_training()
        except Exception as e:
            logger.error(f"💥 Error during training: {e}")
            self.stop_all_training()
            raise
        finally:
            self.print_final_summary()
    
    def stop_all_training(self):
        """Stop all running training processes"""
        logger.info("🛑 Stopping all training processes...")
        self.is_running = False
        
        for trainer in self.trainers:
            trainer.stop()
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=60)
    
    def print_final_summary(self):
        """Print final summary across all GPUs"""
        print(f"\n{Colors.BRIGHT_CYAN}=" * 80)
        print(f"📊 FINAL MULTI-GPU TRAINING SUMMARY")
        print("=" * 80 + Colors.RESET)
        
        total_completed = 0
        total_failed = 0
        
        for trainer in self.trainers:
            completed = len(trainer.completed_trainings)
            failed = len(trainer.failed_trainings)
            total_completed += completed
            total_failed += failed
            
            status_color = Colors.GREEN if failed == 0 else Colors.YELLOW
            print(f"GPU {trainer.gpu_id} ({trainer.dataset}): {status_color}{completed} completed, {failed} failed{Colors.RESET}")
        
        print(f"{Colors.CYAN}-{Colors.RESET}" * 80)
        print(f"📈 Total across all GPUs: {Colors.BRIGHT_GREEN}{total_completed} completed{Colors.RESET}, {Colors.BRIGHT_RED}{total_failed} failed{Colors.RESET}")
        print(f"🎯 Expected total: 16 trainings")
        
        if total_failed == 0 and total_completed == 16:
            print(f"{Colors.BRIGHT_GREEN}🎉 ALL TRAININGS COMPLETED SUCCESSFULLY!{Colors.RESET}")
        elif total_failed == 0:
            print(f"{Colors.YELLOW}⚠️  Some trainings may not have started{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠️  {total_failed} trainings failed{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_CYAN}=" * 80 + Colors.RESET)

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info("⛔ Received interrupt signal, stopping training...")
    sys.exit(0)

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description='Multi-GPU Training Manager for ICICLE Benchmark')
    parser.add_argument('--datasets', nargs=4, 
                       help='Four dataset names (one per GPU). If not provided, interactive mode will be used.')
    parser.add_argument('--config-root', default='/fs/scratch/PAS2099/camera-trap-final/configs',
                       help='Root path to configuration files')
    parser.add_argument('--workspace', default='/fs/ess/PAS2099/sooyoung/ICICLE-Benchmark',
                       help='Workspace path containing run_pipeline.py')
    parser.add_argument('--conda-env', default='icicle',
                       help='Conda environment name to use for training')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running training')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Get datasets - either from command line or interactive selection
        if args.datasets:
            datasets = args.datasets
            print(f"{Colors.GREEN}📝 Using command line datasets: {', '.join(datasets)}{Colors.RESET}")
        else:
            datasets = select_datasets_interactive(args.config_root)
            if not datasets:
                print(f"{Colors.RED}❌ No datasets selected. Exiting.{Colors.RESET}")
                sys.exit(1)
        
        # Create manager
        manager = MultiGPUTrainingManager(
            datasets=datasets,
            config_root_path=args.config_root,
            workspace_path=args.workspace,
            conda_env=args.conda_env
        )
        
        if args.dry_run:
            print(f"\n{Colors.BRIGHT_YELLOW}🔍 Dry Run Mode - Validation Only{Colors.RESET}")
            manager.validate_configs()
            manager.print_training_plan()
            print(f"\n{Colors.GREEN}✅ Configuration validation passed{Colors.RESET}")
        else:
            # Run actual training
            manager.start_training()
            
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()