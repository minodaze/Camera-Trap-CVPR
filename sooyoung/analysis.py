#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/users/PAS2099/lemengwang/Documents/icicle/Camera-Trap-CVPR/sooyoung/merged.csv"   # change if needed

def main():
    # 1. Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        sys.exit(1)

    # 2. Check column
    if "total img" not in df.columns:
        print("Column 'total img' not found in CSV.")
        sys.exit(1)

    # 3. Get numeric values (drop NaNs)
    images = pd.to_numeric(df["total img"], errors="coerce").dropna()
    if images.empty:
        print("No valid numeric values in 'total img'.")
        sys.exit(1)

    # 4. Plot histogram with 100 bins
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(images, bins=100)

    ax.set_xlabel("Image Count")
    ax.set_ylabel("Frequency")

    # 5. Rotate x tick labels 45° (斜着)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    fig.tight_layout()
    plt.show()  # or save with fig.savefig("image_count_hist.png", dpi=300)

if __name__ == "__main__":
    main()

# %%
