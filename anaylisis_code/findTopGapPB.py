import os
import pandas as pd
from glob import glob

base_dir = "/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/np_result_basic"
csv_files = glob(os.path.join(base_dir, "*_per_ckp.csv"))

results = []

for path in csv_files:
    try:
        df = pd.read_csv(path)
        # Ensure only numeric columns are considered
        cols = ["random", "always_yes", "always_no", "pick_higher"]
        means = df[cols].mean()
        
        best_baseline = max(means["random"], means["always_yes"], means["always_no"])
        diff = means["pick_higher"] - best_baseline
        
        dataset_name = os.path.basename(path).replace("_per_ckp.csv", "")
        results.append((dataset_name, diff))
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")

# Sort by difference descending
results.sort(key=lambda x: x[1], reverse=True)

# Display top 10
print("Top 10 datasets by (pick_higher - best_baseline):")
for name, diff in results[:10]:
    print(f"{name:40s} {diff:.4f}")

# Optional: save to CSV
pd.DataFrame(results, columns=["dataset", "diff"]).to_csv(
    os.path.join(base_dir, "summary_pick_higher_diff.csv"), index=False
)
