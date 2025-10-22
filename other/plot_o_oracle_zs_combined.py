import matplotlib.pyplot as plt
def generate_plot():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    def load_with_firstrow_header(path):
        raw = pd.read_csv(path)
        df = raw.copy()
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        return df

    df1 = load_with_firstrow_header("other/CL + Animal Trap - Oracle _= ZS (1).csv")
    df3 = load_with_firstrow_header("other/CL + Animal Trap - Oracle _ ZS (3).csv")

    s1 = pd.to_numeric(df1["oracle - zs"], errors="coerce")
    s3 = pd.to_numeric(df3["oracle - zs"], errors="coerce")

    combined = pd.concat([s1, s3], ignore_index=True).dropna()
    sorted_vals = combined.sort_values(ascending=False).reset_index(drop=True)

    colors = np.where(sorted_vals > 0, "#29A7D8", np.where(sorted_vals < 0, "#d22b2b", "lightgray"))

    plt.figure(figsize=(9, 6), dpi=220)
    plt.bar(np.arange(len(sorted_vals)), sorted_vals.values, color=colors, width=0.9)
    plt.axhline(0, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Dataset Index", fontsize=16, fontweight="bold")
    plt.ylabel("Oracle Accuracy - ZS Accuracy", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("other/original_oracle_vs_zs_combined.png", dpi=220)
    print("Plot saved to other/original_oracle_vs_zs_combined.png")
    plt.close()

if __name__ == '__main__':
    generate_plot()
    plt.show()
