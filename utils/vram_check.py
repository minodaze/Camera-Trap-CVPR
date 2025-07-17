"""
VRAM checking utilities for GPU memory management.

This module provides centralized functions for checking VRAM usage,
applying automatic cache clearing, and logging critical memory states.
"""

import torch


def get_vram_usage():
    """
    Get current VRAM usage in GB.
    
    Returns:
        float: Current reserved VRAM in GB, or 0.0 if CUDA not available
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_reserved() / 1024**3
    return 0.0


def check_vram_and_clean(critical_threshold=39.0, silent_clean_threshold=37.0, context=""):
    """
    Check VRAM usage and apply cleaning if necessary.
    
    Args:
        critical_threshold (float): GB threshold for critical warning and forced cleaning (default: 39.0)
        silent_clean_threshold (float): GB threshold for silent cache clearing (default: 37.0)
        context (str): Optional context string for logging
        
    Returns:
        float: Current VRAM usage in GB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    torch.cuda.synchronize()
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    
    # Critical warning and forced cleaning at 39GB
    if mem_reserved > critical_threshold:
        context_str = f" ({context})" if context else ""
        print(f"🚨 CRITICAL: Very high VRAM usage{context_str}! Reserved: {mem_reserved:.2f}GB - forcing cache clear")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Silent cleaning at 37GB
    elif mem_reserved > silent_clean_threshold:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return mem_reserved


def silent_vram_check():
    """
    Silent VRAM check without any cleaning or warnings.
    
    Returns:
        float: Current VRAM usage in GB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 1024**3


def conditional_cache_clear(threshold=37.0):
    """
    Clear cache only if VRAM usage exceeds threshold.
    
    Args:
        threshold (float): GB threshold for cache clearing (default: 37.0)
        
    Returns:
        bool: True if cache was cleared, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    torch.cuda.synchronize()
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    
    if mem_reserved > threshold:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return True
    
    return False


def maintenance_vram_check(batch_interval=25, threshold=37.0):
    """
    Maintenance VRAM check for periodic cleaning during training loops.
    
    Args:
        batch_interval (int): Check every N batches (default: 25)
        threshold (float): GB threshold for cleaning (default: 30.0)
        
    Returns:
        function: A function that takes batch_idx and performs conditional cleaning
    """
    def check_and_clean(batch_idx):
        if batch_idx % batch_interval == (batch_interval - 1):
            return conditional_cache_clear(threshold)
        return False
    
    return check_and_clean
