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

import torch
from torch import nn
from torch.nn import functional as F

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

def plot_text_F_PCA(model, class_names, args, prefix=''):
    """
    Plot text features from model.head weights in 2D using PCA.
    Each dot represents a class in class_names.
    
    Args:
        model: The classifier model with .head attribute
        class_names: List of class names
        args: Arguments object with save_dir
        prefix: Optional prefix for the filename
    """
    # Check if model has head
    if not hasattr(model, 'head') or model.head is None:
        logging.warning("⚠️ Model does not have a head or head is None. Cannot plot text features.")
        return
    
    logging.info("📊 Plotting text features (head weights) in 2D...")
    
    # Extract text features (head weights)
    with torch.no_grad():
        # Head weights shape: [num_classes, feature_dim]
        text_features = model.head.weight.cpu().numpy()
    
    logging.info(f"📈 Text features shape: {text_features.shape}")
    logging.info(f"🏷️ Number of classes: {len(class_names)}")
    
    # Use PCA to reduce to 2D
    pca = PCA(n_components=2)
    text_features_2d = pca.fit_transform(text_features)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = plt.scatter(text_features_2d[:, 0], text_features_2d[:, 1], 
                         c=range(len(class_names)), cmap='tab20', s=80, alpha=0.7)
    
    # Add class names as labels
    for i, name in enumerate(class_names):
        plt.annotate(name, 
                    (text_features_2d[i, 0], text_features_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Set labels and title
    plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Text Features (Head Weights) - PCA Projection')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class Index')
    
    plt.tight_layout()
    
    # Save the plot
    if hasattr(args, 'save_dir') and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        filename = f'{prefix}text_features_2d.png' if prefix else 'text_features_2d.png'
        save_path = os.path.join(args.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Text features plot saved to: {save_path}")
    
    plt.show()
    
    # Log PCA info
    total_variance = pca.explained_variance_ratio_.sum()
    logging.info(f"📊 PCA explained variance - PC1: {pca.explained_variance_ratio_[0]:.2%}, "
                f"PC2: {pca.explained_variance_ratio_[1]:.2%}, Total: {total_variance:.2%}")
    
    # Also create t-SNE plot if we have enough classes
    if len(class_names) > 3:
        try:
            logging.info("📊 Creating t-SNE visualization...")
            plot_text_F_tsne(model, class_names, args, prefix)
        except Exception as e:
            logging.warning(f"Failed to create t-SNE plot: {e}")
    
    return text_features_2d, pca


def plot_text_F(model, class_names, args, prefix='', perplexity=30):
    """
    Plot text features from model.head weights in 2D using t-SNE.
    Each dot represents a class in class_names.
    
    Args:
        model: The classifier model with .head attribute
        class_names: List of class names
        args: Arguments object with save_dir
        prefix: Optional prefix for the filename
        perplexity: t-SNE perplexity parameter
    """
    # Check if model has head
    if not hasattr(model, 'head') or model.head is None:
        logging.warning("Model does not have a head or head is None. Cannot plot text features.")
        return
    
    # Extract text features (head weights)
    with torch.no_grad():
        text_features = model.head.weight.cpu().numpy()
    
    logging.info(f"Text features shape: {text_features.shape}")
    logging.info(f"Number of classes: {len(class_names)}")
    
    # Use t-SNE to reduce to 2D (adjust perplexity based on number of classes)
    n_classes = len(class_names)
    perplexity = min(perplexity, max(5, n_classes - 1))  # Ensure valid perplexity
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                max_iter=1000, verbose=1)
    text_features_2d = tsne.fit_transform(text_features)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = plt.scatter(text_features_2d[:, 0], text_features_2d[:, 1], 
                         c=range(len(class_names)), cmap='tab20', s=80, alpha=0.7)
    
    # Add class names as labels
    for i, name in enumerate(class_names):
        plt.annotate(name, 
                    (text_features_2d[i, 0], text_features_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Set labels and title
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'Text Features 2D Projection')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class Index')
    
    plt.tight_layout()
    
    # Save the plot
    if hasattr(args, 'save_dir') and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        filename = f'{prefix}text_features_2d_tsne.png' if prefix else 'text_features_2d_tsne.png'
        save_path = os.path.join(args.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Text features t-SNE plot saved to: {save_path}")
    
    plt.show()
    
    return text_features_2d




