import os
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def plot_piechart(class_counts, title, color_mapping, output_path, show_all_legend=False, figsize=(8, 6)):
    """
    Plot a single pie chart for class distribution.
    
    Args:
        class_counts (dict): Dictionary with class names as keys and counts as values
        title (str): Title for the pie chart
        color_mapping (dict): Dictionary mapping class names to colors
        output_path (str): Path to save the plot
        show_all_legend (bool): Whether to show legend for all classes
        figsize (tuple): Figure size (width, height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort classes by count (descending) for better visualization
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_classes:
        # Create empty plot if no data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    classes = [item[0] for item in sorted_classes]
    counts = [item[1] for item in sorted_classes]
    colors = [color_mapping.get(cls, '#cccccc') for cls in classes]
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot pie chart
    wedges, texts, autotexts = ax.pie(
        counts, 
        labels=classes if len(classes) <= 8 else None,  # Only show labels if not too many classes
        colors=colors,
        autopct='%1.1f%%' if len(classes) <= 8 else None,  # Only show percentages if not too many classes
        startangle=90,
        textprops={'fontsize': 8}
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add legend if needed
    if show_all_legend or len(classes) > 8:
        # Create legend with class names and counts
        legend_labels = [f"{cls} ({count})" for cls, count in sorted_classes]
        ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiple_piecharts(piechart_data, output_path, main_title, color_mapping, max_cols=4):
    """
    Plot multiple pie charts in a grid layout.
    
    Args:
        piechart_data (list): List of tuples (class_counts, subtitle) for each pie chart
        output_path (str): Path to save the plot
        main_title (str): Main title for the entire figure
        color_mapping (dict): Dictionary mapping class names to colors
        max_cols (int): Maximum number of columns in the grid
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not piechart_data:
        # Create empty plot if no data
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(main_title)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    n_charts = len(piechart_data)
    
    # Calculate grid dimensions
    n_cols = min(max_cols, n_charts)
    n_rows = math.ceil(n_charts / n_cols)
    
    # Calculate figure size based on number of subplots
    fig_width = n_cols * 4
    fig_height = n_rows * 3.5 + 1  # Extra space for main title
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle single subplot case
    if n_charts == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each pie chart
    for i, (class_counts, subtitle) in enumerate(piechart_data):
        ax = axes[i]
        
        if not class_counts:
            # Empty subplot
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(subtitle, fontsize=10)
            continue
        
        # Sort classes by count (descending)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        classes = [item[0] for item in sorted_classes]
        counts = [item[1] for item in sorted_classes]
        colors = [color_mapping.get(cls, '#cccccc') for cls in classes]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts,
            colors=colors,
            startangle=90,
            textprops={'fontsize': 6}
        )
        
        # Set subtitle
        ax.set_title(subtitle, fontsize=10)
    
    # Hide empty subplots
    for i in range(n_charts, len(axes)):
        axes[i].set_visible(False)
    
    # Set main title
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.95)
    
    # Create a single legend for all pie charts
    # Collect all unique classes from all pie charts
    all_classes = set()
    for class_counts, _ in piechart_data:
        all_classes.update(class_counts.keys())
    
    if all_classes:
        # Sort classes for consistent legend order
        sorted_all_classes = sorted(all_classes)
        colors = [color_mapping.get(cls, '#cccccc') for cls in sorted_all_classes]
        
        # Create legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cls) for cls, color in zip(sorted_all_classes, colors)]
        
        # Add legend to the right side of the figure
        fig.legend(
            handles=legend_elements,
            labels=sorted_all_classes,
            title="Classes",
            loc='center right',
            bbox_to_anchor=(0.98, 0.5),
            fontsize=8
        )
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_color_palette(n_colors):
    """
    Create a color palette with n_colors distinct colors.
    
    Args:
        n_colors (int): Number of colors needed
        
    Returns:
        list: List of color tuples (R, G, B, A)
    """
    # Use matplotlib's tab20, tab20b, tab20c colormaps for up to 60 colors
    colormaps = [plt.cm.get_cmap("tab20"), plt.cm.get_cmap("tab20b"), plt.cm.get_cmap("tab20c")]
    colors = []
    
    for i in range(n_colors):
        colormap = colormaps[i // 20 % len(colormaps)]
        color = colormap(i % 20)
        colors.append(color)
    
    return colors
