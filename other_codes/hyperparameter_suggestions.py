#!/usr/bin/env python3
"""
Hyperparameter suggestions for ICICLE-Benchmark loss functions
"""

import numpy as np

def get_loss_hyperparams(samples_per_cls, loss_type="cdt"):
    """
    Generate recommended hyperparameters based on dataset characteristics
    
    Args:
        samples_per_cls: List of sample counts per class
        loss_type: "focal", "cb_focal", or "cdt"
    
    Returns:
        dict: Recommended hyperparameters
    """
    max_samples = max(samples_per_cls)
    min_samples = min(samples_per_cls)
    imbalance_ratio = max_samples / min_samples
    
    print(f"Dataset Analysis:")
    print(f"  Max samples per class: {max_samples}")
    print(f"  Min samples per class: {min_samples}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    print(f"  Total classes: {len(samples_per_cls)}")
    
    if loss_type == "focal":
        if imbalance_ratio > 1000:
            gamma = 3.0
        elif imbalance_ratio > 100:
            gamma = 2.0
        else:
            gamma = 1.0
            
        # Calculate alpha weights (inverse frequency)
        alpha = np.array([max_samples / n for n in samples_per_cls])
        alpha = alpha / alpha.sum() * len(samples_per_cls)
        
        return {
            "gamma": gamma,
            "alpha": alpha.tolist(),
            "reasoning": f"Gamma={gamma} for imbalance_ratio={imbalance_ratio:.1f}"
        }
    
    elif loss_type == "cb_focal":
        if imbalance_ratio > 1000:
            beta = 0.99999
            gamma = 3.0
        elif imbalance_ratio > 100:
            beta = 0.9999
            gamma = 2.0
        else:
            beta = 0.999
            gamma = 1.0
            
        return {
            "beta": beta,
            "gamma": gamma,
            "loss_type": "focal",
            "reasoning": f"Beta={beta}, Gamma={gamma} for imbalance_ratio={imbalance_ratio:.1f}"
        }
    
    elif loss_type == "cdt":
        if imbalance_ratio > 1000:
            gamma = 0.5
        elif imbalance_ratio > 100:
            gamma = 0.3  # Current setting
        else:
            gamma = 0.1
            
        return {
            "gamma": gamma,
            "reasoning": f"CDT gamma={gamma} for imbalance_ratio={imbalance_ratio:.1f}"
        }
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def hyperparameter_search_grid():
    """
    Generate grid search parameters for different loss functions
    """
    return {
        "focal": {
            "gamma": [0.5, 1.0, 2.0, 3.0, 5.0]
        },
        "cb_focal": {
            "beta": [0.99, 0.999, 0.9999, 0.99999],
            "gamma": [0.5, 1.0, 2.0, 3.0]
        },
        "cdt": {
            "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        }
    }

def analyze_dataset_balance(data_path):
    """
    Analyze class balance in your dataset
    """
    # This would need to be adapted to your specific data format
    print("Dataset balance analysis:")
    print("1. Load your train.json files")
    print("2. Count samples per class")
    print("3. Calculate imbalance metrics")
    print("4. Use get_loss_hyperparams() for recommendations")

if __name__ == "__main__":
    # Example usage with hypothetical data
    # Replace with your actual class distribution
    example_samples = [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]  # Typical long-tail
    
    print("=== Hyperparameter Recommendations ===\n")
    
    for loss_type in ["focal", "cb_focal", "cdt"]:
        print(f"\n--- {loss_type.upper()} Loss ---")
        params = get_loss_hyperparams(example_samples, loss_type)
        for key, value in params.items():
            if key != "reasoning":
                print(f"  {key}: {value}")
        print(f"  Reasoning: {params['reasoning']}")
    
    print("\n=== Grid Search Suggestions ===")
    grid = hyperparameter_search_grid()
    for loss_type, params in grid.items():
        print(f"\n{loss_type}:")
        for param, values in params.items():
            print(f"  {param}: {values}")
