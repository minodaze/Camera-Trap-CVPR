import re, pandas as pd, numpy as np, matplotlib.pyplot as plt

# Load LaTeX table
with open("other/interpolation_table.tex","r",encoding="utf-8") as f:
    tex = f.read()

# Parse rows between \midrule and \bottomrule
block = re.search(r"\\midrule(.*?)\\bottomrule", tex, flags=re.S).group(1)
rows = []
for line in block.splitlines():
    line = line.strip()
    if not line or '...' in line: 
        continue
    if line.endswith(r"\\"): line = line[:-2].strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) != 6: 
        continue
    d, zs, ub, c07, c08, c09 = parts
    rows.append(dict(dataset=d, zs=float(zs), ub=float(ub),
                     a07=float(c07), a08=float(c08), a09=float(c09)))
df = pd.DataFrame(rows)

# Data
x = np.arange(len(df))
zs, ub, a08 = df['zs'].values, df['ub'].values, df['a08'].values
labels = df['dataset'].str.replace('/', '_').values  # prettify labels
labels = [l.split('_')[0] + '_' + l.split('_')[-1] for l in labels]
# Plot
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(x, zs, marker='o', color='green', label='Zero Shot')
# Turn dots into curves for oracle and alpha=0.8
ax.plot(x, ub, marker='s', color='orange', label='Oracle', linestyle='-', linewidth=2)
ax.plot(x, a08, marker='^', color='blue', label=r'Coefficient Weight $\alpha=0.8$', linestyle='-', linewidth=2)

# Arrows oracle → α=0.8 (green if improve, red if worse)
# for xi, y0, y1 in zip(x, ub, a08):
#     color = 'green' if y1 >= y0 else 'red'
#     ax.annotate("", xy=(xi, y1), xytext=(xi, y0),
#                 arrowprops=dict(arrowstyle="->", color=color, lw=1.3, shrinkA=0, shrinkB=0))

# Styling to match your example
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha='right')
ax.set_ylabel("Accuracy per Catergory", fontsize=14)
ax.set_xlabel("Dataset", fontsize=14)

ax.grid(True, linestyle=':', linewidth=0.6)
ax.legend(
    title="Interpolation Coefficient Effect",
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=3,
    frameon=True,
    fontsize=12,          # bigger label text
    title_fontsize=13,    # bigger title text
    markerscale=1.2       # slightly bigger markers in legend
)
fig.tight_layout()
fig.savefig("other/interp_alpha08_arrows.png", dpi=220)