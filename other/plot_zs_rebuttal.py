import pandas as pd
import matplotlib.pyplot as plt


ALL_METRICS_CSV = "/users/PAS2099/mino/ICICLE/other/camera-trap-CVPR - all metrics (1).csv"
ZS_CSV = "/users/PAS2099/mino/ICICLE/other/camera-trap-CVPR - CLIP _ Siglip2 ZS.csv"


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_actual = {str(c).strip().casefold(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().casefold()
        if key in lower_to_actual:
            return lower_to_actual[key]
    return None


def _coerce_score_to_unit_interval(series: pd.Series) -> pd.Series:
    # Handles values like 0.83, 83, "83%", "0.83 ", etc.
    s = series.astype(str).str.strip()
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    # If values look like percentages, convert to [0,1].
    looks_like_percent = out.dropna().gt(1.0).mean() > 0.5
    if looks_like_percent:
        out = out / 100.0
    return out


all_metrics = pd.read_csv(ALL_METRICS_CSV)
zs_raw = pd.read_csv(ZS_CSV)

# Normalize headers (the root cause of the KeyError is typically whitespace/casing mismatches)
all_metrics.columns = all_metrics.columns.astype(str).str.strip()
zs_raw.columns = zs_raw.columns.astype(str).str.strip()

zs_dataset_col = _find_column(zs_raw, ["dataset", "Dataset", "data_set"])
zs_score_col = _find_column(zs_raw, ["BioCLIP2", "BioCLIP 2", "bioclip2", "ZS", "zs"])  # fallbacks

if zs_dataset_col is None or zs_score_col is None:
    raise KeyError(
        "Could not find required columns in ZS CSV. "
        f"Found columns: {list(zs_raw.columns)}; expected something like 'dataset' and 'BioCLIP2'."
    )

zs = zs_raw[[zs_dataset_col, zs_score_col]].rename(columns={zs_dataset_col: "dataset", zs_score_col: "zs"}).copy()
zs["zs"] = _coerce_score_to_unit_interval(zs["zs"])

# If you want *all datasets from all-metrics* (even missing zs), merge here; otherwise plot only those with zs.
if "dataset" in all_metrics.columns:
    merged = all_metrics[["dataset"]].drop_duplicates().merge(zs, on="dataset", how="left")
else:
    merged = zs

plot_df = merged.dropna(subset=["zs"]).drop_duplicates(subset=["dataset"]).copy()

# Group bins: 60-70%, 70-80%, 80-90%, and >= 90%
bins = [0, 0.60, 0.70, 0.80, 0.88, float("inf")]
labels = ["<60%", "60–70%", "70–80%", "80–90%", ">=90%"]
plot_df["zs_bin"] = pd.cut(plot_df["zs"], bins=bins, labels=labels, right=False, include_lowest=True)

counts = plot_df["zs_bin"].value_counts(dropna=False).reindex(labels, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(counts.index.astype(str), counts.values, color="#4C78A8", edgecolor="#2F3B52", linewidth=1.0)

ax.set_title("Zero-shot (ZS) results across datasets")
ax.set_xlabel("ZS shot result range")
ax.set_ylabel("Number of datasets")
ax.grid(axis="y", linestyle="--", alpha=0.35)

for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2.0, h + max(counts.max() * 0.02, 0.2), f"{int(h)}", ha="center", va="bottom", fontsize=10)

out_png = "/users/PAS2099/mino/ICICLE/other/zs_bins_hist.png"
plt.tight_layout()
plt.savefig(out_png, dpi=200)

print("Counts by bin:")
print(counts.to_string())
print(f"\nSaved plot to: {out_png}")
