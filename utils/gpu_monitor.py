"""
GPU Memory Monitoring Utilities

This module provides utilities for monitoring GPU memory usage during training,
with special focus on PETL (Parameter Efficient Transfer Learning) methods.

Features:
- Colored terminal output (enabled by default)
- Aligned column formatting for easy comparison
- PETL-specific parameter tracking
- Wandb integration

Usage Examples:
    
    # Basic usage with colors (default) and aligned output
    from utils.gpu_monitor import get_gpu_monitor, colored_log, Colors
    
    # Initialize monitor (colors enabled by default)
    monitor = get_gpu_monitor(device='cuda', enable_wandb=True)
    
    # Log memory usage (automatically aligned)
    monitor.log_memory_usage("model_load", "after_initialization")
    monitor.log_memory_usage("training", "epoch_1_start")
    
    # Monitor model parameters (especially useful for PETL methods)
    monitor.monitor_model_parameters(model, "my_model")
    
    # Use colored logging
    colored_log("Training started!", Colors.GREEN)
    colored_log("Warning: High memory usage", Colors.YELLOW, "warning")
    colored_log("Error occurred", Colors.RED, "error")
    
    # In training pipeline:
    # Colors enabled by default
    python run_pipeline.py --gpu_memory_monitor --wandb
    
    # Disable colors if needed
    python run_pipeline.py --gpu_memory_monitor --no_gpu_monitor_colors --wandb
"""

import torch
import logging
import time
from typing import Dict, Optional, List, Tuple
import gc

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False

# ANSI color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'  # Reset to default color

def colored_log(message: str, color: str = Colors.BLUE, level: str = "info", enable_colors: bool = True):
    """Log a colored message."""
    if enable_colors:
        colored_message = f"{color}{message}{Colors.RESET}"
    else:
        colored_message = message
        
    if level == "info":
        logging.info(colored_message)
    elif level == "warning":
        logging.warning(colored_message)
    elif level == "error":
        logging.error(colored_message)
    elif level == "debug":
        logging.debug(colored_message)


class GPUMemoryMonitor:
    """Monitor GPU memory usage with detailed PETL parameter tracking."""
    
    def __init__(self, device: str = 'cuda', enable_wandb: bool = False, enable_colors: bool = True):
        """
        Initialize GPU memory monitor.
        
        Args:
            device: Device to monitor (default: 'cuda')
            enable_wandb: Whether to log to wandb
            enable_colors: Whether to use colored output (default: True)
        """
        self.device = device
        self.enable_wandb = enable_wandb and _has_wandb and wandb.run is not None
        self.enable_colors = enable_colors
        self.baseline_memory = 0
        self.measurements = []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.baseline_memory = torch.cuda.memory_allocated(device)
            colored_log("GPU Memory Monitor initialized successfully", Colors.GREEN, enable_colors=self.enable_colors)
        else:
            colored_log("CUDA not available, GPU monitoring disabled", Colors.YELLOW, "warning", enable_colors=self.enable_colors)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics in MB."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2   # MB
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**2  # MB
        free = total_memory - allocated
        
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'free': free,
            'total': total_memory
        }
    
    def log_memory_usage(self, stage: str, step: Optional[str] = None, 
                        extra_info: Optional[Dict] = None):
        """
        Log current memory usage.
        
        Args:
            stage: Stage of training (e.g., 'model_load', 'data_load', 'training')
            step: Optional step information
            extra_info: Additional information to log
        """
        if not torch.cuda.is_available():
            return
            
        stats = self.get_memory_stats()
        step_info = f" - {step}" if step else ""
        
        # Format memory values with consistent width for alignment
        allocated_str = f"{stats['allocated']:8.1f} MB"
        reserved_str = f"{stats['reserved']:8.1f} MB"
        free_str = f"{stats['free']:8.1f} MB"
        usage_str = f"{stats['allocated']/stats['total']*100:5.1f} %"
        
        # Create colored log message with aligned columns
        if self.enable_colors:
            memory_info = (
                f"GPU Memory [{Colors.CYAN}{stage:10}{step_info}{Colors.RESET}]: "
                f"Allocated: {Colors.GREEN}{allocated_str:>10}{Colors.RESET}, "
                f"Reserved: {Colors.YELLOW}{reserved_str:>10}{Colors.RESET}, "
                f"Free: {Colors.BLUE}{free_str:>10}{Colors.RESET} "
                f"({Colors.PURPLE}{usage_str:>6}{Colors.RESET} used )"
            )
        else:
            memory_info = (
                f"GPU Memory [{stage:10}{step_info}]: "
                f"Allocated: {allocated_str:>10}, "
                f"Reserved: {reserved_str:>10}, "
                f"Free: {free_str:>10} "
                f"({usage_str:>6} used )"
            )
        
        # Log to console with aligned formatting
        logging.info(memory_info)
        
        # Log to wandb if enabled
        if self.enable_wandb:
            log_data = {
                f"gpu_memory/{stage}_allocated_mb": stats['allocated'],
                f"gpu_memory/{stage}_reserved_mb": stats['reserved'],
                f"gpu_memory/{stage}_free_mb": stats['free'],
                f"gpu_memory/{stage}_usage_percent": stats['allocated']/stats['total']*100
            }
            
            if extra_info:
                for key, value in extra_info.items():
                    log_data[f"gpu_memory/{stage}_{key}"] = value
                    
            wandb.log(log_data)
        
        # Store measurement
        self.measurements.append({
            'timestamp': time.time(),
            'stage': stage,
            'step': step,
            'stats': stats,
            'extra_info': extra_info or {}
        })
    
    def monitor_model_parameters(self, model: torch.nn.Module, 
                               model_name: str = "model") -> Dict[str, int]:
        """
        Monitor and log model parameter counts by component.
        
        Args:
            model: PyTorch model to analyze
            model_name: Name for logging
            
        Returns:
            Dictionary with parameter counts by component
        """
        param_counts = {}
        total_params = 0
        trainable_params = 0
        
        # Group parameters by module type/name
        module_groups = {
            'petl_adapters': [],
            'petl_lora': [],
            'petl_vpt': [],
            'petl_ssf': [],
            'petl_fact': [],
            'petl_convpass': [],
            'petl_repadapter': [],
            'petl_vqt': [],
            'attention': [],
            'mlp': [],
            'norm': [],
            'embedding': [],
            'head': [],
            'other': []
        }
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            # Categorize parameters
            name_lower = name.lower()
            if 'adapter' in name_lower and 'repadapter' not in name_lower:
                module_groups['petl_adapters'].append((name, param_count, param.requires_grad))
            elif 'lora' in name_lower:
                module_groups['petl_lora'].append((name, param_count, param.requires_grad))
            elif 'vpt' in name_lower or 'prompt' in name_lower:
                module_groups['petl_vpt'].append((name, param_count, param.requires_grad))
            elif 'ssf' in name_lower:
                module_groups['petl_ssf'].append((name, param_count, param.requires_grad))
            elif 'fact' in name_lower:
                module_groups['petl_fact'].append((name, param_count, param.requires_grad))
            elif 'convpass' in name_lower:
                module_groups['petl_convpass'].append((name, param_count, param.requires_grad))
            elif 'repadapter' in name_lower:
                module_groups['petl_repadapter'].append((name, param_count, param.requires_grad))
            elif 'vqt' in name_lower:
                module_groups['petl_vqt'].append((name, param_count, param.requires_grad))
            elif 'attn' in name_lower or 'attention' in name_lower:
                module_groups['attention'].append((name, param_count, param.requires_grad))
            elif 'mlp' in name_lower or 'fc' in name_lower:
                module_groups['mlp'].append((name, param_count, param.requires_grad))
            elif 'norm' in name_lower or 'ln' in name_lower:
                module_groups['norm'].append((name, param_count, param.requires_grad))
            elif 'embed' in name_lower or 'pos' in name_lower:
                module_groups['embedding'].append((name, param_count, param.requires_grad))
            elif 'head' in name_lower or 'classifier' in name_lower:
                module_groups['head'].append((name, param_count, param.requires_grad))
            else:
                module_groups['other'].append((name, param_count, param.requires_grad))
        
        # Calculate statistics for each group
        for group_name, params_list in module_groups.items():
            if params_list:
                group_total = sum(count for _, count, _ in params_list)
                group_trainable = sum(count for _, count, trainable in params_list if trainable)
                param_counts[f"{group_name}_total"] = group_total
                param_counts[f"{group_name}_trainable"] = group_trainable
                
                # Log detailed breakdown for PETL methods
                if group_name.startswith('petl_') and group_trainable > 0:
                    colored_log(f"  {group_name.upper()} Parameters:", Colors.CYAN, enable_colors=self.enable_colors)
                    for param_name, count, trainable in params_list:
                        if trainable:
                            colored_log(f"    {param_name:<40}: {count:>8,} params", Colors.GREEN, enable_colors=self.enable_colors)
        
        param_counts['total_params'] = total_params
        param_counts['trainable_params'] = trainable_params
        param_counts['frozen_params'] = total_params - trainable_params
        
        # Log summary with aligned formatting
        colored_log(f"{model_name} Parameter Summary:", Colors.BOLD + Colors.BLUE, enable_colors=self.enable_colors)
        colored_log(f"  {'Total:':<12} {total_params:>12,} parameters", Colors.WHITE, enable_colors=self.enable_colors)
        colored_log(f"  {'Trainable:':<12} {trainable_params:>12,} parameters ({trainable_params/total_params*100:5.2f}%)", Colors.GREEN, enable_colors=self.enable_colors)
        colored_log(f"  {'Frozen:':<12} {total_params - trainable_params:>12,} parameters", Colors.YELLOW, enable_colors=self.enable_colors)
        
        # Log PETL-specific summary
        petl_total = sum(param_counts.get(f"petl_{method}_trainable", 0) 
                        for method in ['adapters', 'lora', 'vpt', 'ssf', 'fact', 'convpass', 'repadapter', 'vqt'])
        if petl_total > 0:
            colored_log(f"  {'PETL Total:':<12} {petl_total:>12,} trainable parameters ({petl_total/trainable_params*100:5.2f}% of trainable)", Colors.PURPLE, enable_colors=self.enable_colors)
        
        # Log to wandb if enabled
        if self.enable_wandb:
            wandb_data = {f"model_params/{model_name}_{k}": v for k, v in param_counts.items()}
            wandb.log(wandb_data)
        
        return param_counts
    
    def monitor_data_loading(self, dataset_name: str, batch_size: int, 
                           num_batches: int, data_loader = None):
        """
        Monitor memory usage during data loading.
        
        Args:
            dataset_name: Name of the dataset
            batch_size: Batch size
            num_batches: Number of batches
            data_loader: Optional data loader to get sample batch
        """
        colored_log(f"Monitoring data loading for {dataset_name} (batch_size={batch_size}, num_batches={num_batches})", Colors.BLUE, enable_colors=self.enable_colors)
        
        # Before data loading
        self.log_memory_usage("data_load", "before", {
            'dataset': dataset_name,
            'batch_size': batch_size,
            'num_batches': num_batches
        })
        
        # Sample a batch if data_loader is provided
        if data_loader is not None:
            try:
                sample_batch = next(iter(data_loader))
                if isinstance(sample_batch, (list, tuple)):
                    # Calculate approximate batch memory
                    batch_memory = 0
                    for item in sample_batch:
                        if torch.is_tensor(item):
                            batch_memory += item.element_size() * item.nelement()
                    
                    self.log_memory_usage("data_load", "sample_batch", {
                        'dataset': dataset_name,
                        'batch_memory_mb': batch_memory / 1024**2
                    })
                    
                del sample_batch
                torch.cuda.empty_cache()
            except Exception as e:
                colored_log(f"Could not sample batch for memory monitoring: {e}", Colors.RED, "warning", enable_colors=self.enable_colors)
    
    def clear_cache_and_log(self, stage: str):
        """Clear GPU cache and log memory after clearing."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage(stage, "after_cache_clear")
    
    def get_memory_summary(self) -> Dict:
        """Get summary of all memory measurements."""
        if not self.measurements:
            return {}
        
        summary = {
            'total_measurements': len(self.measurements),
            'peak_allocated': max(m['stats']['allocated'] for m in self.measurements),
            'peak_reserved': max(m['stats']['reserved'] for m in self.measurements),
            'stages': list(set(m['stage'] for m in self.measurements))
        }
        
        return summary


# Global monitor instance
_gpu_monitor = None

def get_gpu_monitor(device: str = 'cuda', enable_wandb: bool = False, enable_colors: bool = True) -> GPUMemoryMonitor:
    """Get or create global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMemoryMonitor(device, enable_wandb, enable_colors)
    return _gpu_monitor

def log_gpu_memory(stage: str, step: Optional[str] = None, 
                  extra_info: Optional[Dict] = None, 
                  device: str = 'cuda', enable_wandb: bool = False, enable_colors: bool = True):
    """Convenience function to log GPU memory usage."""
    monitor = get_gpu_monitor(device, enable_wandb, enable_colors)
    monitor.log_memory_usage(stage, step, extra_info)

def monitor_model_memory(model: torch.nn.Module, model_name: str = "model",
                        device: str = 'cuda', enable_wandb: bool = False, enable_colors: bool = True) -> Dict[str, int]:
    """Convenience function to monitor model parameters."""
    monitor = get_gpu_monitor(device, enable_wandb, enable_colors)
    return monitor.monitor_model_parameters(model, model_name)
