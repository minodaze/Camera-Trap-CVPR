import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------- Config ---------
CSV_PATH = "CL + Animal Trap - Final ML Study Dataset (4).csv"

FIRST_FIVE = [
    'orinoquia/orinoquia_N25',
    'na/na_archbold_FL-32',
    'serengeti/serengeti_E08',
    'serengeti/serengeti_K13',
    'serengeti/serengeti_T13',
]

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

OUT1 = "paired_5_arrows.png"
OUT2 = "paired_10_arrows.png"

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
    lo = max(0.0, v.min() - 0.02)
    hi = min(1.0, v.max() + 0.02)
    return lo, hi

lo, hi = global_limits(sub1, sub2)

def plot_pairs(subdf, title, outpath):
    x = subdf['Full FT + CE'].to_numpy(float)
    y_full = subdf['Full FT + BSM'].to_numpy(float)
    y_lora = subdf['LoRA + BSM'].to_numpy(float)
    names = subdf['dataset'].to_numpy(str)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(x, y_full, label='Full FT + BSM', marker='o', color='blue')
    ax.scatter(x, y_lora, label='LoRA + BSM', marker='^', color='blue')

    # arrows with color coding
    for xi, yi1, yi2, name in zip(x, y_full, y_lora, names):
        if not (np.isnan(xi) or np.isnan(yi1) or np.isnan(yi2)):
            color = 'green' if yi2 > yi1 else 'red'
            ax.annotate("",
                        xy=(xi, yi2), xycoords='data',
                        xytext=(xi, yi1), textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            # midpoint label
            y_mid = (yi1 + yi2) / 2
            ax.text(xi + 0.01, y_mid, name, fontsize=8, va='center')

    # 45-degree reference
    ax.plot([lo, hi], [lo, hi], '--', color='black', linewidth=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Full FT + CE')
    ax.set_ylabel('BSM Accuracy')
    ax.set_title(title)
    ax.grid(True, linestyle=':')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    print(f"Saved {outpath}")

# Make the figures
plot_pairs(sub1, 'Paired Results (5 datasets)', OUT1)
plot_pairs(sub2, 'Paired Results (10 datasets)', OUT2)
