import os
import numpy as np
import pandas as pd

root = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/acc_matrices_from_eval_only"

def diag_next_ckp_diffs(df: pd.DataFrame):
    """
    Return list of diagonal next-ckp diffs:
      diff[i] = | M[i, i] - M[i-1, i] |   for i = 1..K-1
    where K = min(nrows, ncols). Row 0 ('zs') has no diff.
    """
    # keep only numeric ckp_* columns in order
    cols = [c for c in df.columns if c.startswith("ckp_")]
    df = df[cols].astype(float)

    nrows, ncols = df.shape
    K = min(nrows, ncols)
    diffs = []
    for i in range(1, K):
        prev_val = df.iat[i-1, i]
        curr_val = df.iat[i,   i]
        if np.isnan(prev_val) or np.isnan(curr_val):
            diffs.append(np.nan)
        else:
            diffs.append(abs(curr_val - prev_val))
    return diffs  # length = K-1

def dataset_score(df: pd.DataFrame) -> float:
    """Score = (sum of diagonal diffs) / (K-1), counting NaN diffs as 0 in numerator."""
    diffs = diag_next_ckp_diffs(df)
    K_minus_1 = max(len(diffs), 1)
    return np.nansum(diffs) / K_minus_1

# Compute and rank
results = {}
for file in os.listdir(root):
    if not file.endswith("_matrix_balanced_accuracy.csv"):
        continue
    path = os.path.join(root, file)
    df = pd.read_csv(path, index_col=0)
    score = dataset_score(df)
    name = file.replace("_matrix_balanced_accuracy.csv", "")
    results[name] = score

top10 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 datasets by diagonal next-ckp |M[i,i]-M[i-1,i]| (avg over i):\n")
for name, val in top10:
    print(f"{name:40s} {val:.6f}")

# Optional: save full ranking
out_path = os.path.join(root, "diag_next_ckp_diff_ranking.csv")
pd.DataFrame(sorted(results.items(), key=lambda x: x[1], reverse=True),
             columns=["dataset", "score"]).to_csv(out_path, index=False)
print(f"\nSaved full ranking to {out_path}")
