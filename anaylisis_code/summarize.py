#!/usr/bin/env python3
import os
import pandas as pd

# ---- paths (EDIT if needed) ----
NP_RESULT_DIR = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_feature_saving_new_threshold"
ALLOW_LIST_CSV = "always_no_pick_higher_nonzero_diffs_feature_saving_new_threshold0.15.csv"
OUT_CSV = "../when_to_adapt_out/summary_needTrain_feature_saving_new_threshold0.15.csv"

# ---- load allow-list: (dataset, ckp) pairs ----
allow_df = pd.read_csv(ALLOW_LIST_CSV)  # columns: dataset, ckp, diff
required_cols = {"dataset", "ckp"}
if not required_cols.issubset(allow_df.columns):
    raise ValueError(f"Allow-list must contain columns: {required_cols}, got {allow_df.columns}")

rows = []

for dataset_name, group in allow_df.groupby("dataset"):
    per_ckp_path = os.path.join(NP_RESULT_DIR, f"{dataset_name}_per_ckp.csv")
    if not os.path.exists(per_ckp_path):
        print(f"[WARN] Missing per-ckp csv for dataset {dataset_name}: {per_ckp_path}")
        continue

    per_ckp_df = pd.read_csv(per_ckp_path)  # has ckp, random, always_yes, ...

    # Make sure ckp is string on both sides
    per_ckp_df["ckp"] = per_ckp_df["ckp"].astype(str)
    group = group.copy()
    group["ckp"] = group["ckp"].astype(str)

    for _, row in group.iterrows():
        ckp = row["ckp"]
        matched = per_ckp_df[per_ckp_df["ckp"] == ckp]

        if matched.empty:
            print(f"[WARN] No row found for dataset={dataset_name}, ckp={ckp} in {per_ckp_path}")
            continue

        m = matched.iloc[0]

        rows.append({
            "dataset": dataset_name,
            "ckp": ckp,
            "random": m["random"],
            "always_yes": m["always_yes"],
            "always_no": m["always_no"],
            "ood_msp": m["ood_msp"],
            "class_mean": m["class_mean"],
            "pick_higher": m["pick_higher"],
        })

# ---- write summary ----
summary_df = pd.DataFrame(rows, columns=[
    "dataset", "ckp",
    "random", "always_yes", "always_no",
    "ood_msp", "class_mean", "pick_higher"
])
summary_df.to_csv(OUT_CSV, index=False)
print(f"Saved summary with {len(summary_df)} rows to {OUT_CSV}")
