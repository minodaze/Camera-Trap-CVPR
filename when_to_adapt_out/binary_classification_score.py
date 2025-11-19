#!/usr/bin/env python3
import pandas as pd
import numpy as np

# --- paths (edit if needed) ---
no_train_path = "summary_noTrain_feature_saving_new_threshold.csv"
need_train_path = "summary_needTrain_feature_saving_new_threshold.csv"

# numeric equality tolerance (floating-point safe compare)
ATOL = 1e-8

def pct_equal_to_pick(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Return % of rows where each method equals pick_higher.
    """
    # Ensure numeric
    for c in ["random", "ood_msp", "class_mean", "pick_higher"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    eq = lambda c: np.isclose(df[c].values, df["pick_higher"].values, atol=ATOL).mean() * 100.0
    return pd.Series(
        {
            "random":     eq("random"),
            "ood_msp":    eq("ood_msp"),
            "class_mean": eq("class_mean"),
        },
        name=label,
    )

def main():
    # Load
    no_train = pd.read_csv(no_train_path)
    need_train = pd.read_csv(need_train_path)

    # --- OPTIONAL: sanity checks for your expectations ---
    # For summary_noTrain.csv: always_no >= always_yes, and always_no == pick_higher
    if all(col in no_train.columns for col in ["always_no", "always_yes", "pick_higher"]):
        a_no  = pd.to_numeric(no_train["always_no"], errors="coerce")
        a_yes = pd.to_numeric(no_train["always_yes"], errors="coerce")
        ph    = pd.to_numeric(no_train["pick_higher"], errors="coerce")

        cond_ge = (a_no >= a_yes).mean() * 100
        cond_eq = np.isclose(a_no, ph, atol=ATOL).mean() * 100
        print(f"[summary_noTrain] always_no ≥ always_yes in {cond_ge:.2f}% rows")
        print(f"[summary_noTrain] always_no == pick_higher in {cond_eq:.2f}% rows")

    # For summary_needTrain.csv: always_yes >= always_no, and always_yes == pick_higher
    if all(col in need_train.columns for col in ["always_no", "always_yes", "pick_higher"]):
        a_no  = pd.to_numeric(need_train["always_no"], errors="coerce")
        a_yes = pd.to_numeric(need_train["always_yes"], errors="coerce")
        ph    = pd.to_numeric(need_train["pick_higher"], errors="coerce")

        cond_ge = (a_yes >= a_no).mean() * 100
        cond_eq = np.isclose(a_yes, ph, atol=ATOL).mean() * 100
        print(f"[summary_needTrain] always_yes ≥ always_no in {cond_ge:.2f}% rows")
        print(f"[summary_needTrain] always_yes == pick_higher in {cond_eq:.2f}% rows")

    # --- your requested 9 percentages ---
    pct_no_train   = pct_equal_to_pick(no_train,   "summary_noTrain")
    pct_need_train = pct_equal_to_pick(need_train, "summary_needTrain")
    pct_combined   = pct_equal_to_pick(
        pd.concat([no_train, need_train], ignore_index=True),
        "combined"
    )

    # Combine into one DataFrame
    out = pd.concat([pct_no_train, pct_need_train, pct_combined], axis=1)

    # Pretty print
    print("\nPercent of rows where method == pick_higher (by set):")
    print(out.round(2).astype(float))

    # Optionally save to CSV
    # out.round(6).to_csv("pick_match_percentages.csv")

if __name__ == "__main__":
    main()
