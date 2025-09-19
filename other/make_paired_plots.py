import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------- Config ---------
CSV_PATH = "other/CL + Animal Trap - Final ML Study Dataset (4).csv"  # put the CSV in the same folder as this script or use an absolute path

# Datasets for the 5-point plot
FIRST_FIVE = [
    'orinoquia/orinoquia_N25',
    'na/na_archbold_FL-32',
    'serengeti/serengeti_E08',
    'serengeti/serengeti_K13',
    'serengeti/serengeti_T13',
]

# Datasets for the 10-point plot
SECOND_TEN = [
    'serengeti/serengeti_E03',
    'KGA/KGA_KHOGB07',
    'nz/nz_EFH_HCAMC01',
    'KGA/KGA_KHOLA08',
    'orinoquia/orinoquia_N29',
    'nz/nz_EFH_HCAMD03',
    'nz/nz_EFH_HCAMB10',
    'KGA/KGA_KHOLA03',
    'nz/nz_PS1_CAM6213',
    'nz/nz_EFD_DCAMB02',
]

# Output paths
OUT1 = "paired_5.png"
OUT2 = "paired_10.png"

# --------- Load CSV ---------
df = pd.read_csv(CSV_PATH)
for col in ['Full FT + CE','Full FT + BSM','LoRA + BSM']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

have = set(df['dataset'])
FIRST_FIVE = [d for d in FIRST_FIVE if d in have]
SECOND_TEN = [d for d in SECOND_TEN if d in have]

sub1 = df[df['dataset'].isin(FIRST_FIVE)].copy()
sub2 = df[df['dataset'].isin(SECOND_TEN)].copy()

def global_limits(*subs):
    vals = []
    for s in subs:
        vals.append(s[['Full FT + CE','Full FT + BSM','LoRA + BSM']].to_numpy().astype(float).ravel())
    v = np.concatenate(vals)
    v = v[~np.isnan(v)]
    lo = max(0.0, v.min() - 0.005)  # Much smaller padding
    hi = min(1.0, v.max() + 0.005)  # Much smaller padding
    return lo, hi

def get_tight_limits(subdf, padding=0.01):
    """Calculate tight limits based on actual data range for better zoom"""
    x = subdf['Full FT + CE'].to_numpy(float)
    y_full = subdf['Full FT + BSM'].to_numpy(float)
    y_lora = subdf['LoRA + BSM'].to_numpy(float)
    
    # Combine all values and remove NaNs
    all_vals = np.concatenate([x, y_full, y_lora])
    all_vals = all_vals[~np.isnan(all_vals)]
    
    if len(all_vals) == 0:
        return 0.0, 1.0
    
    data_min = all_vals.min()
    data_max = all_vals.max()
    data_range = data_max - data_min
    
    # Add small padding
    lo = max(0.0, data_min - data_range * padding)
    hi = min(1.0, data_max + data_range * padding)
    
    return lo, hi

lo, hi = global_limits(sub1, sub2)

def plot_pairs(subdf, title, outpath, use_tight_zoom=True):
    x = subdf['Full FT + CE'].to_numpy(float)
    y_full = subdf['Full FT + BSM'].to_numpy(float)
    y_lora = subdf['LoRA + BSM'].to_numpy(float)
    names = subdf['dataset'].to_numpy(str)

    # Calculate limits - use tight zoom or global limits
    if use_tight_zoom:
        x_lo, x_hi = get_tight_limits(subdf, padding=0.02)
        y_lo, y_hi = get_tight_limits(subdf, padding=0.02)
        
        # Extract actual data ranges for better zoom calculation
        x_vals = x[~np.isnan(x)]
        y_vals = np.concatenate([y_full[~np.isnan(y_full)], y_lora[~np.isnan(y_lora)]])
        
        if len(x_vals) > 0 and len(y_vals) > 0:
            x_range = x_vals.max() - x_vals.min()
            y_range = y_vals.max() - y_vals.min()
            padding_val = 0.02
            
            x_lo = max(0.0, x_vals.min() - x_range * padding_val)
            x_hi = min(1.0, x_vals.max() + x_range * padding_val)
            y_lo = max(0.0, y_vals.min() - y_range * padding_val)
            y_hi = min(1.0, y_vals.max() + y_range * padding_val)
            
            # Make sure we have a reasonable range
            if (x_hi - x_lo) < 0.1:
                center_x = (x_lo + x_hi) / 2
                x_lo = max(0.0, center_x - 0.05)
                x_hi = min(1.0, center_x + 0.05)
            if (y_hi - y_lo) < 0.1:
                center_y = (y_lo + y_hi) / 2
                y_lo = max(0.0, center_y - 0.05)
                y_hi = min(1.0, center_y + 0.05)
        else:
            x_lo, x_hi = 0.0, 1.0
            y_lo, y_hi = 0.0, 1.0
    else:
        x_lo, x_hi = lo, hi
        y_lo, y_hi = lo, hi

    fig, ax = plt.subplots(figsize=(8, 8))  # Larger figure for better visibility
    
    # Larger markers for better visibility
    ax.scatter(x, y_full, label='Full FT + BSM', marker='o', color='blue', s=80, alpha=0.7)
    ax.scatter(x, y_lora, label='LoRA + BSM', marker='^', color='green', s=80, alpha=0.7)

    # arrows with color coding - thicker arrows
    for i, (xi, yi1, yi2, name) in enumerate(zip(x, y_full, y_lora, names)):
        if not (np.isnan(xi) or np.isnan(yi1) or np.isnan(yi2)):
            color = 'darkgreen' if yi2 > yi1 else 'darkred'
            ax.annotate("",
                        xy=(xi, yi2), xycoords='data',
                        xytext=(xi, yi1), textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.5, alpha=0.8))
            
            # Add dataset labels near the points - keep inside plot bounds
            mid_y = (yi1 + yi2) / 2
            offset_x = (x_hi - x_lo) * 0.02
            offset_y = (y_hi - y_lo) * 0.01
            
            # Calculate proposed label position
            label_x = xi + offset_x
            label_y = mid_y + offset_y
            
            # Adjust position to keep label inside plot area
            if label_x > x_hi - (x_hi - x_lo) * 0.15:  # If too far right
                label_x = xi - offset_x  # Put on left side
                ha_align = 'right'
            else:
                ha_align = 'left'
            
            if label_y > y_hi - (y_hi - y_lo) * 0.05:  # If too high
                label_y = mid_y - offset_y
            elif label_y < y_lo + (y_hi - y_lo) * 0.05:  # If too low
                label_y = mid_y + offset_y
            
            # Ensure label stays within bounds
            label_x = max(x_lo + (x_hi - x_lo) * 0.02, min(x_hi - (x_hi - x_lo) * 0.02, label_x))
            label_y = max(y_lo + (y_hi - y_lo) * 0.02, min(y_hi - (y_hi - y_lo) * 0.02, label_y))
            
            ax.text(label_x, label_y, name.split('/')[-1], 
                   fontsize=8, va='center', ha=ha_align,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

    # 45-degree reference line
    ref_lo = max(x_lo, y_lo)
    ref_hi = min(x_hi, y_hi)
    ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], '--', color='gray', linewidth=1.5, alpha=0.8, label='y = x')

    if title == 'Paired Results (oracle > zs)':
        ax.set_xlim(x_lo + 0.2, x_hi)
        ax.set_ylim(y_lo + 0.2, y_hi)
    elif title == 'Paired Results (oracle < zs)':
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    else:
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('Full FT + CE', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy per Category', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, loc='lower right', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
    print(f"Saved {outpath}")

# Make the figures with tight zoom
print("🎯 Creating zoomed paired plots...")
plot_pairs(sub1, 'Paired Results (oracle < zs) - Zoomed', "paired_5_zoomed.png", use_tight_zoom=True)
plot_pairs(sub2, 'Paired Results (oracle > zs) - Zoomed', "paired_10_zoomed.png", use_tight_zoom=True)

# Optional: Also create the original wide-view plots for comparison
print("📊 Creating original wide-view plots...")
plot_pairs(sub1, 'Paired Results (oracle < zs)', OUT1, use_tight_zoom=False)
plot_pairs(sub2, 'Paired Results (oracle > zs)', OUT2, use_tight_zoom=False)

print("✅ All plots generated!")
print("   🔍 *_zoomed.png - Tight zoom on data points")
print("   📊 paired_*.png - Original wide view")