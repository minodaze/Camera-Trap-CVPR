import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 1. Paste your data here ======
raw = """model/test_interval,interval 1,interval 2,interval 3,interval 4,interval 5,interval 6,interval 7,interval 8,interval 9,interval 10,interval 11
zs,1.000000,0.861818,0.940476,0.807143,1.000000,0.841667,0.877778,0.921818,0.925000,0.975000,1.000000
model 1,,0.861818,0.940476,0.807143,1.000000,0.816667,0.877778,0.921818,0.925000,0.975000,1.000000
model 2,,,0.869048,0.645238,1.000000,0.729167,0.720960,0.700000,0.900000,0.809091,1.000000
model 3,,,,0.821429,1.000000,0.904167,0.700253,0.780000,0.825000,0.593182,1.000000
model 4,,,,,1.000000,0.904167,0.728030,0.940000,0.925000,0.750000,1.000000
model 5,,,,,,0.841667,0.891667,0.941818,0.925000,0.950000,1.000000
model 6,,,,,,,0.703283,0.800000,0.825000,0.568182,1.000000
model 7,,,,,,,,0.960000,0.925000,0.750000,0.969697
model 8,,,,,,,,,0.925000,0.925000,1.000000
model 9,,,,,,,,,,0.750000,0.969697
model 10,,,,,,,,,,,1.000000
"""

# ====== 2. Load into a DataFrame ======
df = pd.read_csv(io.StringIO(raw))

# Use 'model' column as index
df = df.set_index("model/test_interval")

# Drop the test column so only intervals are plotted
# df = df.drop(columns=["test_interval"])

# Convert to numeric array
values = df.to_numpy(dtype=float)

# ====== 3. Create heatmap ======
fig, ax = plt.subplots(figsize=(8, 4))

# Mask NaNs so they don't show as colored cells
masked_values = np.ma.masked_invalid(values)

# Colormap: viridis with NaNs as white
cmap = plt.cm.viridis.copy()
cmap.set_bad(color="white")

im = ax.imshow(
    masked_values,
    cmap=cmap,
    aspect="auto",
    vmin=0.6,  # adjust if you want a different scale
    vmax=1.0,
)

# Set ticks/labels
ax.set_xticks(np.arange(df.shape[1]))
ax.set_yticks(np.arange(df.shape[0]))
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.index)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# ====== 4. Add value annotations ======
n_rows, n_cols = values.shape
for i in range(n_rows):
    for j in range(n_cols):
        v = values[i, j]
        if not np.isnan(v):
            # bright background -> dark text, dark background -> light text
            text_color = "white" if v >= 0.9 else "black"
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

ax.set_xlabel("Interval")
ax.set_ylabel("Model")
ax.set_title("Interval-wise Model Selection Accuracy Heatmap - CDB_CDB_A05")

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Balanced Accuracy")

fig.tight_layout()

# Save first, then show
plt.savefig("interval_model_selection_heatmap_CDB_CDB_A05.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()