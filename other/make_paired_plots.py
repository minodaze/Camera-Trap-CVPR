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

# Ensure numeric types for the columns we need
for col in ['Full FT + CE', 'Full FT + BSM', 'LoRA + BSM']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter lists to datasets that actually exist in the CSV
have = set(df['dataset'])
FIRST_FIVE = [d for d in FIRST_FIVE if d in have]
SECOND_TEN = [d for d in SECOND_TEN if d in have]

sub1 = df[df['dataset'].isin(FIRST_FIVE)].copy()
sub2 = df[df['dataset'].isin(SECOND_TEN)].copy()

def global_limits(*subs):
    import numpy as np
    vals = []
    for s in subs:
        vals.append(s[['Full FT + CE','Full FT + BSM','LoRA + BSM']].to_numpy().astype(float).ravel())
    v = np.concatenate(vals)
    v = v[~np.isnan(v)]
    lo = max(0.0, v.min() - 0.02)
    hi = min(1.0, v.max() + 0.02)
    return lo, hi

lo, hi = global_limits(sub1, sub2)

def plot_pairs(subdf, title, outpath):
    x = subdf['Full FT + CE'].to_numpy(float)
    y_full = subdf['Full FT + BSM'].to_numpy(float)
    y_lora = subdf['LoRA + BSM'].to_numpy(float)
    names = subdf['dataset'].to_numpy(str)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(x, y_full, label='Full FT + BSM', marker='o')
    ax.scatter(x, y_lora, label='LoRA + BSM', marker='^')

    # connect the pair and add label
    for xi, yi1, yi2, name in zip(x, y_full, y_lora, names):
        if not (np.isnan(xi) or np.isnan(yi1) or np.isnan(yi2)):
            # vertical connector
            ax.plot([xi, xi], [yi1, yi2], '-', linewidth=0.8, alpha=0.5)
            # place label at the midpoint
            y_mid = (yi1 + yi2) / 2
            ax.text(xi + 0.01, y_mid, name, fontsize=8, va='center')

    # 45-degree reference line
    ax.plot([lo, hi], [lo, hi], '--', linewidth=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Full FT + CE')
    ax.set_ylabel('Accuracy per Category')
    ax.set_title(title)
    ax.grid(True, linestyle=':')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    print(f"Saved {outpath}")

# Make the figures
plot_pairs(sub1, 'Paired Results (oracle < zs)', OUT1)
plot_pairs(sub2, 'Paired Results (oracle > zs)', OUT2)
