#!/usr/bin/env python3
"""
Read merged CSV and plot class distribution as a bar plot (CVPR-styled).

Behavior:
- Tries to read 'mearged.csv' (typo-friendly), falls back to 'merged.csv' in the same folder.
- Expects a column named 'classes'.
- X-axis: every integer class ID from min to max (1 bin per integer).
- Y-axis: count of rows per class (including zeros for missing classes in the range).
- Styling approximates a clean CVPR figure: Times-like font, thin spines, y-grid, tight layout.
- Color palette supports keys: 'zs', 'best-accum', 'accum', 'best-oracle', 'oracle'.
- Saves PNG (300 dpi) and PDF to the same folder. Does not show the plot.
"""

import os
import sys
import warnings
import argparse
import math
import numpy as np

# Use a non-interactive backend to avoid display requirements
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

def find_csv(base_dir: str) -> str:
	"""Return path to mearged.csv if exists, else merged.csv. Raise if none exist."""
	candidates = [
		os.path.join(base_dir, "mearged.csv"),
		os.path.join(base_dir, "merged.csv"),
	]
	for p in candidates:
		if os.path.isfile(p):
			return p
	raise FileNotFoundError("Neither 'mearged.csv' nor 'merged.csv' found in " + base_dir)


def apply_cvpr_style():
	"""Minimal rcParams tweaks; keep default font as requested."""
	matplotlib.rcParams.update({
		"figure.dpi": 100,
		"savefig.dpi": 300,
		# Use globals set below for consistent thickness
		"axes.linewidth": SPINE_LW,
		# Thicker ticks
		"xtick.major.width": TICK_W_MAJOR,
		"ytick.major.width": TICK_W_MAJOR,
		"xtick.minor.width": TICK_W_MINOR,
		"ytick.minor.width": TICK_W_MINOR,
		"xtick.major.size": TICK_LEN_MAJOR,
		"ytick.major.size": TICK_LEN_MAJOR,
		"xtick.minor.size": TICK_LEN_MINOR,
		"ytick.minor.size": TICK_LEN_MINOR,
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	})


PALETTE = {
	"zs": "#98FB98",  # pale green
	"best-accum": "#87CEEB",  # sky blue
	"accum": "#4A93AF",  # deep slate blue
	"best-oracle": "#FFA07A",  # light salmon
	"oracle": "#D68464",  # muted red
}

# Pastel colors for each panel
IMAGE_COLOR = "#EBA1AE"  
CLASS_COLOR = "#A7C7E7"   # pastel blue
MONTHS_COLOR = "#F7C6A3"  # pastel peach/orange
GINI_COLOR = "#CFE8CF"    # pastel green
TEMPORAL_COLOR = "#CAB3C7"  # pastel purple

# Global thickness constants for a slightly bolder CVPR look
SPINE_LW = 2.0
BAR_EDGE_LW = 2.0
TICK_W_MAJOR = 2.0
TICK_W_MINOR = 1.8
TICK_LEN_MAJOR = 8
TICK_LEN_MINOR = 5

# Font sizes (hard-coded controls)
X_TICK_FONTSIZE = 15
Y_TICK_FONTSIZE = 13
X_LABEL_FONTSIZE = 15
Y_LABEL_FONTSIZE = 20

# Label padding (distance between axis labels and tick labels)
X_LABEL_PAD = 8  # horizontal axis label gap
Y_LABEL_PAD = 8  # vertical axis label gap


def parse_args():
	parser = argparse.ArgumentParser(description="Plot class distribution bar chart (CVPR-styled)")
	parser.add_argument(
		"--color-key",
		default="accum",
		choices=list(PALETTE.keys()),
		help="Which named color to use for bars",
	)
	parser.add_argument(
		"--title",
		default="Class Distribution",
		help="Custom figure title",
	)
	parser.add_argument(
		"--out-name",
		default=None,
		help="Base output filename without extension. Default: classes_count_bar_[color-key]",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	apply_cvpr_style()
	here = os.path.dirname(os.path.abspath(__file__))
	try:
		csv_path = find_csv(here)
	except FileNotFoundError as e:
		print(str(e))
		sys.exit(1)

	# Read CSV
	try:
		df = pd.read_csv(csv_path)
	except Exception as e:
		print(f"Failed to read CSV at {csv_path}: {e}")
		sys.exit(1)

	if "classes" not in df.columns:
		print("Column 'classes' not found in CSV.")
		sys.exit(1)

	# Coerce to numeric (integers) and drop NaNs
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		classes = pd.to_numeric(df["classes"], errors="coerce")
	classes = classes.dropna()
	if classes.empty:
		print("No valid numeric values in 'classes' column after coercion.")
		sys.exit(1)

	# Ensure integer class IDs
	classes = classes.astype(int)

	# Create full index from min to max (per-class, no binning)
	min_c, max_c = int(classes.min()), int(classes.max())
	full_index = pd.Index(range(min_c, max_c + 1), name="class")

	counts = classes.value_counts().sort_index()
	counts = counts.reindex(full_index, fill_value=0)

	# Plot
	n_bins = len(full_index)
	# For paper figures: aim for ~3.3 to 7 inches width depending on bins
	fig_w = max(3.3, min(7.0, n_bins * 0.12))
	fig_h = 2.4 if n_bins <= 30 else 2.8 if n_bins <= 80 else 3.0
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	# Pastel fill with black edge, no gaps between bars
	pastel_color = CLASS_COLOR
	x = list(range(n_bins))
	# Center bars on integer positions so tick labels align under bar centers
	bars = ax.bar(x, counts.values, width=1.0, align="center", color=pastel_color, edgecolor="black", linewidth=BAR_EDGE_LW)
	# Remove left/right padding and ensure first/last bars touch plot edges
	ax.margins(x=0)
	ax.set_xlim(-0.5, n_bins - 0.5)

	# X ticks: place at every odd class ID; thin if too many to avoid overlap (no rotation)
	ticks_text = counts.index.astype(str).tolist()
	class_ids = counts.index.tolist()
	odd_positions = [i for i, cid in enumerate(class_ids) if int(cid) % 2 == 1]
	if not odd_positions:
		# Fallback: if no odd labels present, keep ~<=20 evenly spaced ticks
		step = max(1, (n_bins + 19) // 20)
		odd_positions = list(range(0, n_bins, step))

	max_ticks = 20
	if len(odd_positions) > max_ticks:
		step = max(1, len(odd_positions) // max_ticks)
		thinned = odd_positions[::step]
		if thinned[-1] != odd_positions[-1]:
			thinned.append(odd_positions[-1])
		tick_positions = thinned
	else:
		tick_positions = odd_positions

	ax.set_xticks(tick_positions)
	ax.set_xticklabels([ticks_text[i] for i in tick_positions], rotation=0, fontsize=X_TICK_FONTSIZE)

	# Axis labels per request
	ax.set_xlabel("Class Count", fontsize=X_LABEL_FONTSIZE, labelpad=X_LABEL_PAD)
	ax.set_ylabel("# of Camera Traps", fontsize=Y_LABEL_FONTSIZE, labelpad=Y_LABEL_PAD)
	# No plot title per request

	# Clean spines, no y-grid per request for a cleaner style
	for spine in ["top", "right"]:
		ax.spines[spine].set_visible(False)
	# Thicken visible spines
	for spine in ["left", "bottom"]:
		ax.spines[spine].set_linewidth(SPINE_LW)
	# Thicken ticks explicitly
	ax.tick_params(axis="both", which="both", width=TICK_W_MINOR)
	ax.tick_params(axis="both", which="major", length=TICK_LEN_MAJOR, width=TICK_W_MAJOR)
	ax.tick_params(axis="both", which="minor", length=TICK_LEN_MINOR, width=TICK_W_MINOR)
	# Control tick label font sizes
	ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
	ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
	ax.set_axisbelow(True)

	# Y axis ticks/labels
	ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
	ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

	# No per-bar count labels per request

	fig.tight_layout()

	base_out = args.out_name or f"classes_count_bar_pastel"
	png_path = os.path.join(here, base_out + ".png")
	pdf_path = os.path.join(here, base_out + ".pdf")

	try:
		fig.savefig(png_path, dpi=300, bbox_inches="tight")
		fig.savefig(pdf_path, bbox_inches="tight")
	except Exception as e:
		print(f"Failed to save figures: {e}")
		sys.exit(1)
	finally:
		plt.close(fig)

	# Minimal confirmation
	print(f"Saved: {png_path}\nSaved: {pdf_path}")
	# Prepare Time Span (Months) binned by 5
	if "Time Span (Months)" not in df.columns:
		print("Column 'Time Span (Months)' not found in CSV.")
		sys.exit(1)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		months = pd.to_numeric(df["Time Span (Months)"], errors="coerce")
	months = months.dropna()
	if months.empty:
		print("No valid numeric values in 'Time Span (Months)' after coercion.")
		sys.exit(1)
	months = months.astype(int)
	min_m, max_m = int(months.min()), int(months.max())
	# Define right-edge bins at 6, 12, 18, ... up to >= max_m
	start_edge = 6
	edges = []
	e = start_edge
	while e <= max_m:
		edges.append(e)
		e += 6
	if not edges:
		edges = [start_edge]
	elif edges[-1] < max_m:
		# Ensure the last bin includes the actual maximum (e.g., 68 when last multiple is 66)
		edges.append(max_m)
	# Count values within each right-edge bin: [6] then (7..12], (13..18], ...
	counts_vals = []
	for i, R in enumerate(edges):
		if R == start_edge:
			L = start_edge
		else:
			L = R - 5
		mask = (months >= L) & (months <= R)
		counts_vals.append(int(mask.sum()))
	counts_m = pd.Series(counts_vals, index=pd.Index(edges, name="months_right_edge"))

	# Prepare gini_index distribution with right-edge bins at step 0.1
	if "gini_index" not in df.columns:
		print("Column 'gini_index' not found in CSV.")
		sys.exit(1)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		gini = pd.to_numeric(df["gini_index"], errors="coerce")
	gini = gini.dropna()
	if gini.empty:
		print("No valid numeric values in 'gini_index' after coercion.")
		sys.exit(1)
	step = 0.05
	min_g, max_g = float(gini.min()), float(gini.max())
	# Start gini bins aligned to data min (rounded up to step)
	start_edge_g = math.ceil(min_g / step) * step
	edges_g = []
	e = start_edge_g
	while e <= max_g + 1e-9 and len(edges_g) < 10000:
		edges_g.append(round(e, 4))
		e += step
	if not edges_g:
		edges_g = [round(start_edge_g, 4)]
	elif edges_g[-1] < max_g - 1e-9:
		edges_g.append(round(max_g, 4))
	first_left = round(edges_g[0] - step, 4)
	bins = [first_left] + edges_g
	cats = pd.cut(gini, bins=bins, right=True, include_lowest=True)
	counts_g = cats.value_counts(sort=False)
	counts_g.index = pd.Index(edges_g, name="gini_right_edge")

	# Prepare Image Count distribution with right-edge bins
	if "total img" not in df.columns:
		print("Column 'total img' not found in CSV.")
		sys.exit(1)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		images = pd.to_numeric(df["total img"], errors="coerce")
	images = images.dropna()
	if images.empty:
		print("No valid numeric values in 'total img' after coercion.")
		sys.exit(1)
	# n_bins_img = 10
	# min_i, max_i = int(images.min()), int(images.max())
	# img_bins = list(
    #         np.linspace(min_i, max_i, num=n_bins_img + 1, endpoint=True)
    # )
	# img_edges = [int(round(b)) for b in img_bins[1:]]  # right edges
	# img_cats = pd.cut(images, bins=img_bins, right=True, include_lowest=True)
	# counts_img = img_cats.value_counts(sort=False)
	# counts_img.index = pd.Index(img_edges, name="images_right_edge")
	# Use fixed 6 groups:
	# [0,3000], (3000,6000], (6000,9000], (9000,16000], (16000,32000], (32000, max]
	min_i = int(images.min())
	max_i = int(images.max())

	left0 = 0  # start from 0 for image counts (adjust if you really want min_i)
	fixed_edges = [left0, 3000, 6000, 9000, 16000, 32000, max_i]

	# Make sure edges are strictly increasing (in case max_i <= 32000)
	fixed_edges = sorted(set(fixed_edges))
	if len(fixed_edges) < 2:
		print("Not enough distinct edges to form bins for 'total img'.")
		sys.exit(1)

	img_bins = fixed_edges  # 7 edges -> 6 bins
	img_edges = img_bins[1:]  # right edges for labeling: 3000..max_i

	img_cats = pd.cut(images, bins=img_bins, right=True, include_lowest=True)
	counts_img = img_cats.value_counts(sort=False)
	counts_img.index = pd.Index(img_edges, name="images_right_edge")

	# Prepare Temporal Shift distribution with right-edge bins
	if "l1_test" not in df.columns:
		print("Column 'l1_test' not found in CSV.")
		sys.exit(1)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		temporal = pd.to_numeric(df["l1_test"], errors="coerce")
	temporal = temporal.dropna()
	if temporal.empty:
		print("No valid numeric values in 'l1_test' after coercion.")
		sys.exit(1)

	vals = temporal.to_numpy(dtype=float)
	min_t, max_t = float(vals.min()), float(vals.max())
	if max_t == min_t:
		print("All 'l1_test' values are identical; cannot scale to [0,1].")
		sys.exit(1)

	# --- Normalize to [0, 1] ---
	temporal_norm = (vals - min_t) / (max_t - min_t)
	temporal_norm = pd.Series(temporal_norm, index=temporal.index)

	# --- Bin normalized values into 10 bins over [0, 1] ---
	n_bins_tv = 10
	tv_bins = list(np.linspace(0.0, 1.0, num=n_bins_tv + 1, endpoint=True))
	tv_edges = [round(b, 4) for b in tv_bins[1:]]  # right edges 0.1, 0.2, ..., 1.0

	tv_cats = pd.cut(temporal_norm, bins=tv_bins, right=True, include_lowest=True)
	counts_tv = tv_cats.value_counts(sort=False)
	counts_tv.index = pd.Index(tv_edges, name="temporal_right_edge")


	# Three-row layout with uniform axis sizes: 3 columns per row.
	# Row 0 (top): dummy | dummy | dummy  (absorbs any top clipping)
	# Row 1: placeholder | classes | months
	# Row 2: temporal placeholder | dummy | gini
	# All active axes share identical width/height; blank spacer centers row 2.
	from matplotlib.gridspec import GridSpec
	cols = 3
	row_h = fig_h
	# Increase total height and add a top dummy row to push content down
	total_h = row_h * 3 + 2.0
	# Use same width for each column; overall width scales with single panel width
	per_panel_w = max(3.3, min(7.0, fig_w))
	total_w = per_panel_w * cols + 1.2  # small padding
	fig = plt.figure(figsize=(total_w, total_h))
	# Increase hspace between rows and leave extra top space to avoid y-label clipping
	gs = GridSpec(3, cols, figure=fig, hspace=0.9, wspace=0.45)
	# Top dummy row
	ax_top_dummy1 = fig.add_subplot(gs[0,0])
	ax_top_dummy2 = fig.add_subplot(gs[0,1])
	ax_top_dummy3 = fig.add_subplot(gs[0,2])
	# Content rows
	ax_placeholder_main = fig.add_subplot(gs[1,0])
	ax_classes = fig.add_subplot(gs[1,1])
	ax_months = fig.add_subplot(gs[1,2])
	ax_temporal_placeholder = fig.add_subplot(gs[2,0])
	ax_dummy = fig.add_subplot(gs[2,1])  # new dummy plot between temporal and gini
	ax_gini = fig.add_subplot(gs[2,2])

	def draw_panel(ax, counts, xlabel: str, tick_mode: str, color: str):
		n_bins = len(counts)
		x = list(range(n_bins))
		if tick_mode in ("edge6", "gini_edge", "image_count"):
			# Draw bars left-aligned with width=1 and ticks at right edge (x+1)
			ax.bar(x, counts.values, width=1.0, align="edge", color=color, edgecolor="black", linewidth=BAR_EDGE_LW)
			ax.set_xlim(0, n_bins)
		else:
			ax.bar(x, counts.values, width=1.0, align="center", color=color, edgecolor="black", linewidth=BAR_EDGE_LW)
			ax.set_xlim(-0.5, n_bins - 0.5)
		ax.margins(x=0)

		max_ticks = 20
		if tick_mode == "edge6":
			# X ticks at right edges (x+1), labels are the right-edge values (6,12,18,...)
			right_edge_values = counts.index.astype(int).tolist()
			positions = [i + 1 for i in range(n_bins)]
			positions = positions[::2]  # Keep every other tick
			if len(positions) > max_ticks:
				step = max(1, len(positions) // max_ticks)
				positions = positions[::step]
			labels = [str(right_edge_values[i-1]) for i in positions]
			ax.set_xticks(positions)
			ax.set_xticklabels(labels, 
					  rotation=0, 
					# rotation=45, ha="right",
					  fontsize=X_TICK_FONTSIZE)
		elif tick_mode == "gini_edge":
			# X ticks at right edges (x+1), show only odd tenths: 0.1, 0.3, 0.5, ...
			right_edge_values = counts.index.tolist()
			all_positions = [i + 1 for i in range(n_bins)]
			positions = []
			labels_vals = []
			for i, v in enumerate(right_edge_values):
				tenth = v * 10
				r = round(tenth)
				# keep values near multiples of 0.1 AND with odd tenth index (exclude 0.0)
				if abs(tenth - r) < 1e-8 and (r % 2 == 1):
					positions.append(all_positions[i])
					labels_vals.append(v)
			# Thin if still too many
			if len(positions) > max_ticks:
				thin_step = max(1, len(positions) // max_ticks)
				positions = positions[::thin_step]
				labels_vals = labels_vals[::thin_step]
			labels = [f"{v:.1f}" for v in labels_vals]
			ax.set_xticks(positions)
			ax.set_xticklabels(labels, 
					  rotation=0, 
					# rotation=45, ha="right",
					  fontsize=X_TICK_FONTSIZE)
		elif tick_mode == "image_count":
			 # Fixed 6 labels (last one not tied to the largest bin edge)
			desired_vals = [3000, 6000, 9000, 16000, 32000, 80000]

			# Use up to n_bins labels (in case something weird happens)
			n_labels = min(len(desired_vals), n_bins)

			# Evenly spaced positions in index space: 1, 2, ..., n_labels
			positions = [i + 1 for i in range(n_labels)]

			# Format as 3k, 6k, 9k, 16k, 32k, 80k
			def fmt_k(v: int) -> str:
				# assume all are multiples of 1000
				return f"{v // 1000}k" if v >= 1000 else str(v)

			labels = [fmt_k(v) for v in desired_vals[:n_labels]]

			ax.set_xticks(positions)
			ax.set_xticklabels(labels,
					  		# rotation=0,
							rotation=45, ha="right",
							fontsize=X_TICK_FONTSIZE)
			# # 6 bins -> 6 ticks, one at each bin's right edge
			# right_edge_values = counts.index.to_list()   # [3000, 6000, 9000, 16000, 32000, max_i]
			# positions = [i + 1 for i in range(n_bins)]   # ticks at x+1 since bars are align="edge"

			# ax.set_xticks(positions)
			# ax.set_xticklabels(
			# 	[f"{v:,}" for v in right_edge_values],
			# 	rotation=45, ha="right", fontsize=X_TICK_FONTSIZE
			# )
		else:
			# Odd-tick mode for per-class plot
			ticks_text = counts.index.astype(str).tolist()
			ids = counts.index.tolist()
			odd_positions = [i for i, cid in enumerate(ids) if int(cid) % 2 == 1]
			if not odd_positions:
				step = max(1, (n_bins + 19) // 20)
				odd_positions = list(range(0, n_bins, step))
			if len(odd_positions) > max_ticks:
				step = max(1, len(odd_positions) // max_ticks)
				thinned = odd_positions[::step]
				if thinned[-1] != odd_positions[-1]:
					thinned.append(odd_positions[-1])
				tick_positions = thinned
			else:
				tick_positions = odd_positions
			ax.set_xticks(tick_positions)
			ax.set_xticklabels([ticks_text[i] for i in tick_positions], 
					#   rotation=0, 
					rotation=45, ha="right",
					  fontsize=X_TICK_FONTSIZE)

		ax.set_xlabel(xlabel, fontsize=X_LABEL_FONTSIZE, labelpad=X_LABEL_PAD)
		# y-label only on the left (shared y)
		# handled after both panels are drawn

		# Clean spines
		for spine in ["top", "right"]:
			ax.spines[spine].set_visible(False)
		for spine in ["left", "bottom"]:
			ax.spines[spine].set_linewidth(SPINE_LW)
		ax.tick_params(axis="both", which="both", width=TICK_W_MINOR)
		ax.tick_params(axis="both", which="major", length=TICK_LEN_MAJOR, width=TICK_W_MAJOR)
		ax.tick_params(axis="both", which="minor", length=TICK_LEN_MINOR, width=TICK_W_MINOR)
		# Control tick label font sizes per-axis
		ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
		ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
		# Ensure y ticks show as integer values
		ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
		ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

	def draw_placeholder(ax):
		# Empty placeholder matching style
		ax.set_xlim(0, 1)
		ax.set_xticks([])
		ax.set_xlabel("Image Count", fontsize=X_LABEL_FONTSIZE, labelpad=X_LABEL_PAD)
		# y-axis styling
		for spine in ["top", "right"]:
			ax.spines[spine].set_visible(False)
		for spine in ["left", "bottom"]:
			ax.spines[spine].set_linewidth(SPINE_LW)
		ax.tick_params(axis="both", which="both", width=TICK_W_MINOR)
		ax.tick_params(axis="both", which="major", length=TICK_LEN_MAJOR, width=TICK_W_MAJOR)
		ax.tick_params(axis="both", which="minor", length=TICK_LEN_MINOR, width=TICK_W_MINOR)
		# Control tick label font sizes
		ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
		ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
		ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
		ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
		# Optional placeholder text
		ax.text(0.5, 0.5, "Placeholder", ha="center", va="center", fontsize=X_LABEL_FONTSIZE, color="#666", transform=ax.transAxes)

	def draw_top_dummy(ax):
		# Minimal dummy to occupy space without labels
		ax.set_xlim(0, 1)
		ax.set_xticks([])
		ax.set_yticks([])
		for spine in ax.spines.values():
			spine.set_visible(False)

	ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
	ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

	# Draw top dummy row
	draw_top_dummy(ax_top_dummy1)
	draw_top_dummy(ax_top_dummy2)
	draw_top_dummy(ax_top_dummy3)

	# Draw row1 (first content row)
	# draw_placeholder(ax_placeholder_main)
	draw_panel(ax_placeholder_main, counts_img, "Image Count", tick_mode="image_count", color=IMAGE_COLOR)
	ax_placeholder_main.set_ylabel("# of Camera Traps", fontsize=Y_LABEL_FONTSIZE, labelpad=Y_LABEL_PAD)
	draw_panel(ax_classes, counts, "Class Count", tick_mode="odd", color=CLASS_COLOR)
	draw_panel(ax_months, counts_m, "Time Span (Months)", tick_mode="edge6", color=MONTHS_COLOR)
	# Row2 (second content row): temporal placeholder | dummy | gini
	# draw_placeholder(ax_temporal_placeholder)
	draw_panel(ax_temporal_placeholder, counts_tv, "Temporal Shift", tick_mode="gini_edge", color=TEMPORAL_COLOR)
	ax_temporal_placeholder.set_xlabel("Temporal Shift", fontsize=X_LABEL_FONTSIZE, labelpad=X_LABEL_PAD)
	# Add y-axis label for second row in combined figure
	ax_temporal_placeholder.set_ylabel("# of Camera Traps", fontsize=Y_LABEL_FONTSIZE, labelpad=Y_LABEL_PAD)
	draw_placeholder(ax_dummy)
	ax_dummy.set_xlabel("Dummy", fontsize=X_LABEL_FONTSIZE, labelpad=X_LABEL_PAD)
	draw_panel(ax_gini, counts_g, "Gini Index", tick_mode="gini_edge", color=GINI_COLOR)
	# Ensure y tick labels visible on all active data axes
	for ax_active in [ax_classes, ax_months, ax_temporal_placeholder, ax_dummy, ax_gini]:
		ax_active.yaxis.set_tick_params(labelleft=True)
	# Harmonize y-limits across panels for visual consistency
	global_max = max(counts.max(), counts_m.max(), counts_g.max())
	y_lim = int(global_max * 1.05)
	for ax_all in [ax_top_dummy1, ax_top_dummy2, ax_top_dummy3, ax_placeholder_main, ax_classes, ax_months, ax_temporal_placeholder, ax_dummy, ax_gini]:
		ax_all.set_ylim(0, y_lim)
		if ax_all in [ax_placeholder_main]:
			ax_all.set_ylim(0,counts_img.max()*1.05)
		if ax_all in [ax_temporal_placeholder]:
			ax_all.set_ylim(0,counts_tv.max()*1.05)

	fig.tight_layout()
	# Adjust margins: substantially more top space to avoid clipping first-row labels
	fig.subplots_adjust(left=0.12, top=0.995, bottom=0.14)

	base_out = args.out_name or f"placeholder_classes_timespan_temporal_gini_bar_pastel"
	png_path = os.path.join(here, base_out + ".png")
	pdf_path = os.path.join(here, base_out + ".pdf")
	try:
		fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
		fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
	except Exception as e:
		print(f"Failed to save combined figure: {e}")
	else:
		print(f"Saved combined: {png_path}\nSaved combined: {pdf_path}")
	finally:
		plt.close(fig)

	# --- Separate Row 1 Figure (placeholder | classes | months) ---
	fig_r1 = plt.figure(figsize=(per_panel_w * 3 + 0.8, row_h + 1.0))
	gs_r1 = plt.GridSpec(1, 3, figure=fig_r1, wspace=0.45)
	ax_r1_placeholder = fig_r1.add_subplot(gs_r1[0,0])
	ax_r1_classes = fig_r1.add_subplot(gs_r1[0,1])
	ax_r1_months = fig_r1.add_subplot(gs_r1[0,2])
	draw_placeholder(ax_r1_placeholder)
	ax_r1_placeholder.set_ylabel("# of Camera Traps", fontsize=Y_LABEL_FONTSIZE, labelpad=Y_LABEL_PAD)
	draw_panel(ax_r1_classes, counts, "Class Count", tick_mode="odd", color=CLASS_COLOR)
	draw_panel(ax_r1_months, counts_m, "Time Span (Months)", tick_mode="edge6", color=MONTHS_COLOR)
	for a in [ax_r1_placeholder, ax_r1_classes, ax_r1_months]:
		a.set_ylim(0, y_lim)
	fig_r1.tight_layout()
	# More top space for standalone first-row figure
	fig_r1.subplots_adjust(left=0.12, top=0.995, bottom=0.16)
	row1_base = base_out + "_row1"
	png_r1 = os.path.join(here, row1_base + ".png")
	pdf_r1 = os.path.join(here, row1_base + ".pdf")
	try:
		fig_r1.savefig(png_r1, dpi=300, bbox_inches="tight", pad_inches=0.2)
		fig_r1.savefig(pdf_r1, bbox_inches="tight", pad_inches=0.2)
	except Exception as e:
		print(f"Failed to save row1 figure: {e}")
	else:
		print(f"Saved row1: {png_r1}\nSaved row1: {pdf_r1}")
	finally:
		plt.close(fig_r1)

	# --- Separate Row 2 Figure ((spacer) | temporal placeholder | gini) ---
	fig_r2 = plt.figure(figsize=(per_panel_w * 3 + 0.8, row_h + 0.8))
	gs_r2 = plt.GridSpec(1, 3, figure=fig_r2, wspace=0.45)
	ax_r2_temporal = fig_r2.add_subplot(gs_r2[0,0])
	ax_r2_dummy = fig_r2.add_subplot(gs_r2[0,1])
	ax_r2_gini = fig_r2.add_subplot(gs_r2[0,2])
	draw_placeholder(ax_r2_temporal)
	ax_r2_temporal.set_ylabel("# of Camera Traps", fontsize=Y_LABEL_FONTSIZE, labelpad=Y_LABEL_PAD)
	ax_r2_temporal.set_xlabel("Temporal Shift", fontsize=X_LABEL_FONTSIZE, labelpad=X_LABEL_PAD)
	draw_placeholder(ax_r2_dummy)
	ax_r2_dummy.set_xlabel("Dummy", fontsize=X_LABEL_FONTSIZE, labelpad=X_LABEL_PAD)
	draw_panel(ax_r2_gini, counts_g, "Gini Index", tick_mode="gini_edge", color=GINI_COLOR)
	for a in [ax_r2_temporal, ax_r2_dummy, ax_r2_gini]:
		a.set_ylim(0, y_lim)
	fig_r2.tight_layout()
	# Keep consistent top margin across separate row figures (not strictly needed but uniform)
	fig_r2.subplots_adjust(left=0.12, top=0.995, bottom=0.16)
	row2_base = base_out + "_row2"
	png_r2 = os.path.join(here, row2_base + ".png")
	pdf_r2 = os.path.join(here, row2_base + ".pdf")
	try:
		fig_r2.savefig(png_r2, dpi=300, bbox_inches="tight", pad_inches=0.2)
		fig_r2.savefig(pdf_r2, bbox_inches="tight", pad_inches=0.2)
	except Exception as e:
		print(f"Failed to save row2 figure: {e}")
	else:
		print(f"Saved row2: {png_r2}\nSaved row2: {pdf_r2}")
	finally:
		plt.close(fig_r2)
	print(f"Saved: {png_path}\nSaved: {pdf_path}")


if __name__ == "__main__":
	main()

