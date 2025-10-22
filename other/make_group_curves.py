import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------- Config ---------
CSV_PATH = "other/CL + Animal Trap - Final ML Study Dataset (4).csv"  # adjust if stored elsewhere

# Datasets for the 5-point plot
FIRST_FIVE = [
    'orinoquia/orinoquia_N25',
    'na/na_archbold_FL-32',
    'serengeti/serengeti_E08',
    'serengeti/serengeti_K13',
    'serengeti/serengeti_T13',
]

# Datasets for the 10-point plot
SECOND_TEN = [
    'serengeti/serengeti_E03',
    'KGA/KGA_KHOGB07',
    'nz/nz_EFH_HCAMC01',
    'KGA/KGA_KHOLA08',
    'orinoquia/orinoquia_N29',
    'nz/nz_EFH_HCAMD03',
    'nz/nz_EFH_HCAMB10',
    'KGA/KGA_KHOLA03',
    'nz/nz_PS1_CAM6213',
    'nz/nz_EFD_DCAMB02',
]

OUT_5 = "other/group_curves_5.png"
OUT_10 = "other/group_curves_10.png"

COLUMNS = ['Full FT + CE', 'Full FT + BSM', 'LoRA + BSM']
COLORS = {
    'Full FT + CE':  "#7a3699",  # blue
    'Full FT + BSM': "#e1c98c",  # orange
    'LoRA + BSM':    "#52a69b",  # green
}
MARKERS = {
    'Full FT + CE':  'o',
    'Full FT + BSM': 's',
    'LoRA + BSM':    '^',
}


def _short_label(ds: str) -> str:
    """Convert 'a/b_c_d' -> 'a_d' for concise tick labels."""
    parts = ds.replace('/', '_').split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[-1]}"
    return ds


def _load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _prepare_group(df: pd.DataFrame, group: list[str]) -> tuple[np.ndarray, dict, list[str]]:
    """Return x positions, series dict for each column, and x tick labels preserving given order."""
    # Filter by availability
    available = [d for d in group if d in set(df['dataset'])]
    sub = df[df['dataset'].isin(available)].copy()
    # Preserve order
    sub['__order'] = sub['dataset'].apply(lambda d: available.index(d))
    sub = sub.sort_values('__order')

    x = np.arange(len(sub))
    labels = [_short_label(d) for d in sub['dataset'].tolist()]

    series = {}
    for col in COLUMNS:
        series[col] = sub[col].to_numpy(dtype=float)
    return x, series, labels


def _plot_group(x: np.ndarray, series: dict, labels: list[str], title: str, out_path: str):
    plt.figure(figsize=(12, 5.5), dpi=140)

    # Plot each curve with markers
    for col in COLUMNS:
        y = series[col]
        plt.plot(
            x, y,
            label=col,
            color=COLORS[col],
            marker=MARKERS[col],
            linewidth=2.2,
            markersize=6,
            alpha=0.95,
        )

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Balanced Accuracy', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    plt.legend(loc='best', frameon=True)

    # y-limits with gentle padding based on valid values
    all_vals = np.concatenate([series[c] for c in COLUMNS])
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size:
        lo, hi = all_vals.min(), all_vals.max()
        pad = max(0.01, (hi - lo) * 0.08)
        plt.ylim(max(0.0, lo - pad), min(1.0, hi + pad))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main():
    df = _load_df(CSV_PATH)

    # FIRST_FIVE
    x5, series5, labels5 = _prepare_group(df, FIRST_FIVE)
    _plot_group(x5, series5, labels5, 'Oracle < Zero Shot', OUT_5)

    # SECOND_TEN
    x10, series10, labels10 = _prepare_group(df, SECOND_TEN)
    _plot_group(x10, series10, labels10, 'Oracle > Zero Shot', OUT_10)


if __name__ == '__main__':
    main()
