#!/usr/bin/env python3
import pandas as pd
import numpy as np

# --- paths (edit if needed) ---
no_train_path = "summary_noTrain_feature_saving_new_threshold0.15.csv"
need_train_path = "summary_needTrain_feature_saving_new_threshold0.15.csv"

# --- sampling config ---
RANDOM_SEED = 42  # set None for non-reproducible sampling

# numeric equality tolerance (floating-point safe compare)
ATOL = 1e-8

# methods to show in tables (order matters)
METHODS = ["random", "ood_msp", "class_mean", "always_yes", "always_no"]

def ensure_numeric_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric Series for column `name` (NaNs if missing)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([np.nan] * len(df))

def pct_equal_to_pick(df: pd.DataFrame, label: str) -> pd.Series:
    """
    % of rows where each method's achieved accuracy equals pick_higher (within ATOL).
    Includes random, ood_msp, class_mean, always_yes, always_no.
    """
    ph = ensure_numeric_series(df, "pick_higher")
    out = {}
    for m in METHODS:
        col = ensure_numeric_series(df, m)
        out[m] = np.isclose(col.values, ph.values, atol=ATOL).mean()
    return pd.Series(out, name=label)

def mean_acc_with_oracle(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Mean balanced accuracy for methods + 'oracle' (mean of pick_higher).
    """
    out = {m: ensure_numeric_series(df, m).mean() for m in METHODS}
    out["oracle"] = ensure_numeric_series(df, "pick_higher").mean()
    return pd.Series(out, name=label)

def main():
    # Load
    no_train = pd.read_csv(no_train_path)
    need_train_full = pd.read_csv(need_train_path)

    # # --- downsample need_train to EXACTLY len(no_train) rows ---
    # target = len(no_train)
    # if len(need_train_full) < target:
    #     raise ValueError(
    #         f"{need_train_path} ({len(need_train_full)} rows) has fewer rows than {no_train_path} ({target})."
    #     )
    # need_train = (
    #     need_train_full.sample(n=target, replace=False, random_state=RANDOM_SEED)
    #     .reset_index(drop=True)
    # )
    # print(f"[sampling] {need_train_path}: took {len(need_train)} of {len(need_train_full)} rows "
    #       f"to match {no_train_path} ({target}) (seed={RANDOM_SEED})")

        # --- downsample need_train to EXACTLY len(no_train) rows,
    #     choosing largest |always_yes - always_no| (no NaNs assumed) ---
    target = len(no_train)
    if len(need_train_full) < target:
        raise ValueError(
            f"{need_train_path} ({len(need_train_full)} rows) has fewer rows than {no_train_path} ({target})."
        )

    RANDOM_SEED = 42  # reproducible tie-breaker
    ay = need_train_full["always_yes"].to_numpy(dtype=float)
    an = need_train_full["always_no"].to_numpy(dtype=float)
    abs_gap = np.abs(ay - an)

    rng = np.random.RandomState(RANDOM_SEED)
    tie = rng.rand(len(need_train_full))  # random jitter for stable ranking

    need_train = (
        need_train_full
        .assign(abs_gap=abs_gap, __tie=tie)
        .sort_values(by=["abs_gap", "__tie"], ascending=[False, True], kind="mergesort")
        .head(target)
        .drop(columns=["abs_gap", "__tie"])
        .reset_index(drop=True)
    )

    print(f"[selection] {need_train_path}: took top-{target} by |always_yes - always_no| (seed={RANDOM_SEED})")


    # --- OPTIONAL sanity checks (kept) ---
    if all(col in no_train.columns for col in ["always_no", "always_yes", "pick_higher"]):
        a_no  = pd.to_numeric(no_train["always_no"], errors="coerce")
        a_yes = pd.to_numeric(no_train["always_yes"], errors="coerce")
        ph    = pd.to_numeric(no_train["pick_higher"], errors="coerce")
        print(f"[summary_noTrain] always_no ≥ always_yes in {((a_no >= a_yes).mean()*100):.2f}% rows")
        print(f"[summary_noTrain] always_no == pick_higher in {(np.isclose(a_no, ph, atol=ATOL).mean()*100):.2f}% rows")

    if all(col in need_train.columns for col in ["always_no", "always_yes", "pick_higher"]):
        a_no  = pd.to_numeric(need_train["always_no"], errors="coerce")
        a_yes = pd.to_numeric(need_train["always_yes"], errors="coerce")
        ph    = pd.to_numeric(need_train["pick_higher"], errors="coerce")
        print(f"[summary_needTrain (sampled)] always_yes ≥ always_no in {((a_yes >= a_no).mean()*100):.2f}% rows")
        print(f"[summary_needTrain (sampled)] always_yes == pick_higher in {(np.isclose(a_yes, ph, atol=ATOL).mean()*100):.2f}% rows")

    # --- (1) Binary match percentages (methods + always_yes/no) ---
    pct_no_train   = pct_equal_to_pick(no_train,   "summary_noTrain_binary")
    pct_need_train = pct_equal_to_pick(need_train, "summary_needTrain_binary (sampled)")
    pct_combined   = pct_equal_to_pick(pd.concat([no_train, need_train], ignore_index=True), "combined_binary")
    pct_out = pd.concat([pct_no_train, pct_need_train, pct_combined], axis=1)

    print("\nPercent of rows where method == pick_higher (by set):")
    print(pct_out.round(2).astype(float))

    # --- (2) Mean balanced accuracy for methods + oracle IN THE SAME TABLE ---
    acc_no_train   = mean_acc_with_oracle(no_train,   "summary_noTrain_acc")
    acc_need_train = mean_acc_with_oracle(need_train, "summary_needTrain_acc (sampled)")
    acc_combined   = mean_acc_with_oracle(pd.concat([no_train, need_train], ignore_index=True), "combined_acc")

    acc_out = pd.concat([acc_no_train, acc_need_train, acc_combined], axis=1)

    print("\nMean balanced accuracy (methods + always_yes/no + oracle) by set:")
    print(acc_out.round(4).astype(float))

    # Optional saves
    # pct_out.round(6).to_csv("pick_match_percentages.csv")
    # acc_out.round(6).to_csv("method_mean_accuracies_with_oracle.csv")

if __name__ == "__main__":
    main()
