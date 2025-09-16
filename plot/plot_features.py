import argparse
import code
import logging
import os
import copy
import time
import json
import re
import random

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

from datetime import datetime
import numpy as np
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader, Subset
import pickle
import wandb  # Ensure wandb is imported
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from core import *
from core.module import get_al_module, get_cl_module, get_ood_module
from core.calibration import (
    comprehensive_paper_calibration, 
    get_training_classes_for_checkpoint,
    convert_numpy_types
)
from utils.misc import method_name
from utils.gpu_monitor import get_gpu_monitor, log_gpu_memory, monitor_model_memory
from utils.log_formatter import (
    setup_colored_logging, log_section_start, log_subsection_start, log_step,
    log_success, log_warning, log_error, log_info, log_config_item,
    log_checkpoint, log_final_result, log_metric, create_info_box, Colors,
    configure_colors_for_wandb
)

# Add required imports for plotting
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_features(model, dataset, args, method='tsne', n_samples_per_class=100, prefix=''):
    """
    Create a 2D visualization of feature representations with unique colors for each class.
    
    Args:
        model: The trained classifier model
        dataset: Dataset containing samples to visualize
        args: Arguments object containing device and other settings
        method: Dimensionality reduction method ('tsne' or 'pca')
        n_samples_per_class: Maximum number of samples per class to plot
        save_path: Path to save the plot (optional)
    """
    
    logging.info(f"🎨 Creating 2D feature visualization using {method.upper()}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Sample balanced data from each class
    class_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        class_to_indices[sample.label].append(idx)
    
    # Select balanced samples from each class
    selected_indices = []
    for class_label, indices in class_to_indices.items():
        # Randomly sample up to n_samples_per_class from each class
        n_samples = min(n_samples_per_class, len(indices))
        sampled_indices = np.random.choice(indices, n_samples, replace=False)
        selected_indices.extend(sampled_indices)
    
    # Create subset dataset
    subset_dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(subset_dataset, batch_size=128, shuffle=False, num_workers=4)

    logging.info(f"📊 Extracting features from {len(selected_indices)} samples across {len(class_to_indices)} classes")
    
    # Extract features and labels
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, _, _, _) in enumerate(dataloader):
            images = images.to(args.device)
            
            # Extract features using the model's forward_features method
            if hasattr(model, 'forward_features'):
                features = model.forward_features(images)
            else:
                # Fallback: use the visual model directly
                features = model.visual_model(images)
                features = torch.nn.functional.normalize(features, dim=-1)
            
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels.numpy())
            del images, labels, features
            
            if batch_idx % 10 == 0:
                logging.info(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.array(labels_list)
    
    # Safety checks
    if len(all_features) < 3:
        logging.warning(f"⚠️  Insufficient data for visualization: only {len(all_features)} samples found")
        return None, None
    
    if all_features.shape[1] < 2:
        logging.warning(f"⚠️  Insufficient feature dimensions for visualization: only {all_features.shape[1]} dimensions")
        return None, None
    
    logging.info(f"🔧 Applying {method.upper()} dimensionality reduction...")
    logging.info(f"  Input shape: {all_features.shape} (samples x features)")
    
    # Apply dimensionality reduction
    try:
        if method.lower() == 'tsne':
            # Use PCA first to reduce dimensions if needed, then t-SNE to 2D
            max_components = min(50, all_features.shape[1], all_features.shape[0] - 1)
            if all_features.shape[1] > max_components and max_components > 2:
                pca_pre = PCA(n_components=max_components)
                features_reduced = pca_pre.fit_transform(all_features)
                logging.info(f"  Pre-processing with PCA: {all_features.shape[1]} -> {max_components} dimensions")
            else:
                features_reduced = all_features
                
            # Ensure perplexity is valid for t-SNE
            perplexity = min(30, max(5, (len(all_features) - 1) // 3))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            features_2d = reducer.fit_transform(features_reduced)
            logging.info(f"  t-SNE perplexity set to: {perplexity}")
        else:  # PCA
            # Ensure we don't request more components than possible
            max_components = min(2, all_features.shape[1], all_features.shape[0] - 1)
            reducer = PCA(n_components=max_components)
            features_2d = reducer.fit_transform(all_features)
            
            # If we only got 1 component, pad with zeros for the second dimension
            if features_2d.shape[1] == 1:
                features_2d = np.column_stack([features_2d, np.zeros(features_2d.shape[0])])
                logging.info("  Added zero-padding for second PCA component")
                
    except Exception as e:
        logging.error(f"❌ Dimensionality reduction failed: {str(e)}")
        return None, None
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Generate unique colors for each class
    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Create scatter plot for each class
    for i, class_label in enumerate(unique_labels):
        mask = all_labels == class_label
        class_features = features_2d[mask]
        class_name = dataset.class_names[class_label] if class_label < len(dataset.class_names) else f"Class_{class_label}"
        
        plt.scatter(class_features[:, 0], class_features[:, 1], 
                   c=[colors[i]], label=class_name, alpha=0.7, s=20)
    
    plt.title(f'2D Feature Visualization using {method.upper()}\n({len(all_features)} samples, {len(unique_labels)} classes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    
    # Add legend (if not too many classes)
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        plt.text(0.02, 0.98, f'{len(unique_labels)} classes total', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if path provided
    if args.save_dir:
        save_path = os.path.join(args.save_dir, prefix + f'features_{method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"💾 Plot saved to: {save_path}")
    
    plt.show()
    
    # logging.info some statistics
    logging.info(f"\n📈 Visualization Statistics:")
    logging.info(f"  Total samples plotted: {len(all_features)}")
    logging.info(f"  Number of classes: {len(unique_labels)}")
    logging.info(f"  Original feature dimension: {all_features.shape[1]}")
    logging.info(f"  Samples per class: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    
    if method.lower() == 'pca':
        explained_variance = reducer.explained_variance_ratio_
        logging.info(f"  PCA explained variance: {explained_variance[0]:.3f}, {explained_variance[1]:.3f} (total: {sum(explained_variance):.3f})")
    
    return features_2d, all_labels


def plot_features_for_checkpoint(model, dataset, args, ckp_name, method='tsne', output_dir='feature_plots'):
    """
    Convenience function to plot features for a specific checkpoint.
    
    Args:
        model: The trained classifier model
        dataset: Dataset containing samples to visualize
        args: Arguments object containing device and other settings
        ckp_name: Name of the checkpoint (for saving)
        method: Dimensionality reduction method ('tsne' or 'pca')
        output_dir: Directory to save the plots
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate save path
    save_path = os.path.join(output_dir, f'{ckp_name}_features_{method}.png')
    
    logging.info(f"\n🎯 Plotting features for checkpoint: {ckp_name}")
    
    # Call the main plotting function
    features_2d, labels = plot_features(model, dataset, args, method=method, save_path=save_path)
    
    return features_2d, labels


def compare_features_across_checkpoints(models_dict, dataset, args, method='tsne', output_dir='feature_plots'):
    """
    Create subplot comparison of features across multiple checkpoints.
    
    Args:
        models_dict: Dictionary of {checkpoint_name: model} pairs
        dataset: Dataset containing samples to visualize
        args: Arguments object containing device and other settings
        method: Dimensionality reduction method ('tsne' or 'pca')
        output_dir: Directory to save the plots
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_models = len(models_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    logging.info(f"\n🔄 Comparing features across {n_models} checkpoints using {method.upper()}")
    
    for idx, (ckp_name, model) in enumerate(models_dict.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        logging.info(f"  Processing checkpoint: {ckp_name}")
        
        # Plot features for this checkpoint (without showing individual plot)
        features_2d, labels = plot_features(model, dataset, args, method=method, save_path=None)
        
        # Generate colors
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # Plot on subplot
        for i, class_label in enumerate(unique_labels):
            mask = labels == class_label
            class_features = features_2d[mask]
            ax.scatter(class_features[:, 0], class_features[:, 1], 
                      c=[colors[i]], alpha=0.7, s=15)
        
        ax.set_title(f'{ckp_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.suptitle(f'Feature Evolution Across Checkpoints ({method.upper()})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    save_path = os.path.join(output_dir, f'features_comparison_{method}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logging.info(f"💾 Comparison plot saved to: {save_path}")
    
    return save_path
    
