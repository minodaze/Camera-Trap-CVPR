
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_interpolation_curves(csv_path, out_path="other/interpolation_zoomed_plot.png",
                              y_min=0.6, y_max=0.95, title="Upper-bound → Interpolated → Zero-shot"):
    # Load
    df = pd.read_csv(csv_path)

    # Coefficient columns (sorted numerically by alpha)
    coef_cols = sorted([c for c in df.columns if c.startswith("coefficient-")],
                       key=lambda s: float(s.split("-")[-1]))
    alphas = [float(c.split("-")[-1]) for c in coef_cols]

    # Remove datasets with any NaN across the coefficient columns
    mask_no_nan = ~df[coef_cols].isna().any(axis=1)
    plot_df = df.loc[mask_no_nan].copy()

    # Plot
    plt.figure(figsize=(14, 8))
    for _, row in plot_df.iterrows():
        y = row[coef_cols].to_numpy(dtype=float)
        label = row['dataset']
        label = label.replace('/', '_')
        label = label.split('_')[0] + '_' + label.split('_')[-1]
        plt.plot(alphas, y, marker='o', linewidth=2, markersize=6, label=label)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Interpolation coefficient (α)", fontsize=12)
    plt.ylabel("Balanced Accuracy", fontsize=12)
    plt.xlim(min(alphas), max(alphas))
    
    # Tighter y-limits for better zoom
    ymin = max(y_min, plot_df[coef_cols].min().min() - 0.01)
    ymax = min(y_max, plot_df[coef_cols].max().max() + 0.01)
    plt.ylim(ymin, ymax)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Legend outside on the right
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

def plot_interpolation_curves_auto_zoom(csv_path, out_path="other/interpolation_auto_zoomed_plot.png",
                                       zoom_padding=0.05, title="Oracle → Interpolated → Zero-shot"):
    """
    Plot with automatic zoom that finds the optimal range based on data.
    
    Args:
        csv_path: Path to CSV file
        out_path: Output path for the plot
        zoom_padding: Padding around the data range (e.g., 0.05 = 5% padding)
        title: Plot title
    """
    # Load
    df = pd.read_csv(csv_path)

    # Coefficient columns (sorted numerically by alpha)
    coef_cols = sorted([c for c in df.columns if c.startswith("coefficient-")],
                       key=lambda s: float(s.split("-")[-1]))
    alphas = [float(c.split("-")[-1]) for c in coef_cols]

    # Remove datasets with any NaN across the coefficient columns
    mask_no_nan = ~df[coef_cols].isna().any(axis=1)
    plot_df = df.loc[mask_no_nan].copy()

    # Calculate optimal zoom range
    all_values = plot_df[coef_cols].values.flatten()
    all_values = all_values[~np.isnan(all_values)]  # Remove any remaining NaNs
    
    data_min = np.min(all_values)
    data_max = np.max(all_values)
    data_range = data_max - data_min
    
    # Add padding
    y_min_auto = data_min - (data_range * zoom_padding)
    y_max_auto = data_max + (data_range * zoom_padding)
    
    print(f"📊 Auto-zoom range: {y_min_auto:.3f} to {y_max_auto:.3f}")
    print(f"📈 Data range: {data_min:.3f} to {data_max:.3f}")

    # Plot
    plt.figure(figsize=(14, 8))
    for _, row in plot_df.iterrows():
        y = row[coef_cols].to_numpy(dtype=float)
        label = row['dataset']
        label = label.replace('/', '_')
        label = label.split('_')[0] + '_' + label.split('_')[-1]
        plt.plot(alphas, y, marker='o', linewidth=2, markersize=6, label=label)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Interpolation coefficient (α)", fontsize=12)
    plt.ylabel("Balanced Accuracy", fontsize=12)
    plt.xlim(min(alphas), max(alphas))
    plt.ylim(y_min_auto, y_max_auto)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Legend outside on the right
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved auto-zoomed figure to: {out_path}")

def plot_interpolation_curves_focus_region(csv_path, out_path="other/interpolation_focus_plot.png",
                                          focus_y_min=0.7, focus_y_max=0.9, 
                                          title="Upper-bound → Interpolated → Zero-shot (Focus Region)"):
    """
    Plot focusing on a specific region of interest.
    """
    # Load
    df = pd.read_csv(csv_path)

    # Coefficient columns (sorted numerically by alpha)
    coef_cols = sorted([c for c in df.columns if c.startswith("coefficient-")],
                       key=lambda s: float(s.split("-")[-1]))
    alphas = [float(c.split("-")[-1]) for c in coef_cols]

    # Remove datasets with any NaN across the coefficient columns
    mask_no_nan = ~df[coef_cols].isna().any(axis=1)
    plot_df = df.loc[mask_no_nan].copy()

    # Filter datasets that have data in the focus region
    focus_mask = plot_df[coef_cols].apply(
        lambda row: any((focus_y_min <= val <= focus_y_max) for val in row if not np.isnan(val)), 
        axis=1
    )
    focus_df = plot_df.loc[focus_mask].copy()
    
    print(f"📊 Focusing on {len(focus_df)} datasets in range {focus_y_min:.3f} to {focus_y_max:.3f}")

    # Plot
    plt.figure(figsize=(14, 8))
    for _, row in focus_df.iterrows():
        y = row[coef_cols].to_numpy(dtype=float)
        label = row['dataset']
        label = label.replace('/', '_')
        label = label.split('_')[0] + '_' + label.split('_')[-1]
        plt.plot(alphas, y, marker='o', linewidth=2.5, markersize=7, label=label)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Interpolation coefficient (α)", fontsize=12)
    plt.ylabel("Balanced Accuracy", fontsize=12)
    plt.xlim(min(alphas), max(alphas))
    plt.ylim(focus_y_min, focus_y_max)
    
    # Add more detailed grid in focus region
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Legend outside on the right
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, ncol=1, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved focus region plot to: {out_path}")

if __name__ == "__main__":
    csv_file = "other/CL + Animal Trap - Final ML Study Dataset with interpolation_table.csv"
    
    print("🎯 Generating multiple zoom levels...")
    
    # 1. Standard plot with manual zoom (tighter than before)
    print("\n1️⃣ Creating manually zoomed plot...")
    plot_interpolation_curves(csv_file, 
                             out_path="other/interpolation_manual_zoomed_plot.png",
                             y_min=0.6, y_max=0.95)
    
    # 2. Auto-zoom plot (finds optimal range automatically)
    print("\n2️⃣ Creating auto-zoomed plot...")
    plot_interpolation_curves_auto_zoom(csv_file, 
                                       out_path="other/interpolation_auto_zoomed_plot.png",
                                       zoom_padding=0.03)
    
    # 3. Focus region plot (very tight zoom on specific range)
    print("\n3️⃣ Creating focus region plot...")
    plot_interpolation_curves_focus_region(csv_file,
                                          out_path="other/interpolation_focus_plot.png",
                                          focus_y_min=0.75, focus_y_max=0.92)
    
    print("\n✅ All plots generated! Check the 'other/' directory for:")
    print("   📊 interpolation_manual_zoomed_plot.png - Manual zoom")
    print("   📊 interpolation_auto_zoomed_plot.png - Auto zoom") 
    print("   📊 interpolation_focus_plot.png - Focus region")
