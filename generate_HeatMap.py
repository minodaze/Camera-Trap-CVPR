import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Data
methods = ["Random", "MSP Based", "Feature Based", "Always Adapt", "Always Skip", "Oracle"]
no_train   = [0.8016, 0.7275, 0.7417, 0.6533, 0.8624, 0.8624]
need_train = [0.7463, 0.7914, 0.7858, 0.8957, 0.6461, 0.8957]
combined   = [0.7739, 0.7595, 0.7637, 0.7745, 0.7543, 0.8791]

data = np.array([no_train, need_train, combined]).T  # shape (6, 3)
cols = ["Skip", "Adapt", "Combined"]

# === Custom colormap using #4A93AF ===
high_color = "#4A93AF"
cmap = LinearSegmentedColormap.from_list("custom_teal", ["#FFFFFF", high_color])

# Plot
fig, ax = plt.subplots(figsize=(5, 3))

# Heatmap: higher value = darker red
im = ax.imshow(
    data,
    cmap=cmap,          # color map (higher -> more red)
    aspect="auto",
    vmin=data.min(),
    vmax=data.max(),
)

# Ticks and labels
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(methods)))
ax.set_xticklabels(cols, fontweight='bold', fontsize=8)
ax.set_yticklabels(methods, fontweight='bold', fontsize=8)

# Rotate x tick labels a bit for readability
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

# Annotate each cell with its numeric value
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(
            j, i,
            f"{data[i, j]:.4f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            fontweight='bold'
        )

# ax.set_title("Balanced Accuracy of Adaptation Policies")
cbar = fig.colorbar(im, ax=ax)
# Set tick label size
cbar.ax.tick_params(labelsize=8)

# Make tick labels bold
for tick_label in cbar.ax.get_yticklabels():
    tick_label.set_fontweight('bold')
# Set colorbar label
cbar.set_label("Accuracy", rotation=270, labelpad=15, fontsize=8, fontweight='bold')

plt.tight_layout()

# Save high-res figure
plt.savefig("when2adapt_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
