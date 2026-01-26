import pandas as pd

accum_path = "/users/PAS2099/mino/ICICLE/other/CVPR - camera-trap rebuttal - accum dataset.csv"
all_path   = "/users/PAS2099/mino/ICICLE/other/camera-trap-CVPR - all metrics.csv"

alldata = pd.read_csv(all_path)
accum   = pd.read_csv(accum_path)

# normalize column names
alldata.columns = alldata.columns.str.strip()
accum.columns   = accum.columns.str.strip()

# pick the correct training-img column from alldata
train_col = "training imgs" if "training imgs" in alldata.columns else "training img"
best_accum_col = "best accum" if "best accum" in accum.columns else "best accum"  # adjust if your column name differs

accum["dataset_key"] = accum["dataset"].str.replace("_", "/", n=1)

merged = accum.merge(
    alldata[["dataset", train_col]].rename(columns={"dataset": "dataset_key"}),
    on="dataset_key",
    how="left"
)

# numeric cleanup
merged[train_col] = pd.to_numeric(merged[train_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
merged[best_accum_col] = pd.to_numeric(merged[best_accum_col], errors="coerce")

# keep only the range of interest
df = merged.dropna(subset=[best_accum_col]).copy()
df = df[(df[best_accum_col] >= 0.7) & (df[best_accum_col] <= 1.0)]

# bins + quotas
bins   = [0.7, 0.8, 0.9, 1.0000001]
labels = ["0.7-0.8", "0.8-0.9", "0.9-1.0"]
quota  = {"0.7-0.8": 3, "0.8-0.9": 4, "0.9-1.0": 3}

df["accum_bin"] = pd.cut(df[best_accum_col], bins=bins, labels=labels, right=False)

# Randomly select datasets with training imgs < 7000 in each bin
seed = 42
df_small = df.dropna(subset=[train_col]).copy()
df_small = df_small[df_small[train_col] < 7000]

picked_parts = []
for b in labels:
    g = df_small[df_small["accum_bin"] == b]
    n = quota[b]
    if len(g) >= n:
        picked_parts.append(g.sample(n=n, random_state=seed))
    else:
        # Not enough rows under 7000; take what we have and warn.
        print(f"[WARN] Bin {b}: only {len(g)} datasets with {train_col} < 7000 (need {n}).")
        picked_parts.append(g)

picked = pd.concat(picked_parts, ignore_index=True)

print(picked[["dataset", 'best oracle', best_accum_col, train_col, "accum_bin"]])
picked.to_csv("/users/PAS2099/mino/ICICLE/other/picked_10_diverse_best_accum.csv", index=False)