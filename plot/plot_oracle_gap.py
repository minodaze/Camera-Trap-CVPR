import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
try:  # optional dependency for axis breaks
    from brokenaxes import brokenaxes  # type: ignore
    _HAVE_BROKEN_AXES = True
except Exception:
    _HAVE_BROKEN_AXES = False

with open('/users/PAS2099/mino/ICICLE/plot/oracle.txt', 'r') as file:
    datasets = file.read().splitlines()

dataset_num_samples = {}
eval_results = {}
train_results = {}
for dataset in datasets:
    data_path = dataset.replace("_", "/", 1)
    train_path = f"/fs/scratch/PAS2099/camera-trap-benchmark/dataset/{data_path}/30/train.json"
    train_result_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/eval_oracle_20_on_train/oracle_eval_on_train/{dataset}/bioclip2/full_text_head_eval_on_train/log/eval_only_summary.json"
    eval_result_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/oracle/{dataset}/bioclip2/full_text_head/all/log/final_training_summary.json"
    number_samples = 0
    try:
        with open(train_path, 'r') as file:
            data = json.load(file)
        for ckp, samples in data.items():
            number_samples += len(samples)
        dataset_num_samples[dataset] = number_samples
        with open(eval_result_path, 'r') as file:
            eval_data = json.load(file)
        eval_results[dataset] = eval_data['averages']['accuracy']
        with open(train_result_path, 'r') as file:
            train_data = json.load(file)
        train_results[dataset] = train_data['accuracy']
    except FileNotFoundError:
        print(f"File not found for dataset: {dataset}")
        print(f"  Missing train path: {train_result_path} {os.path.exists(train_result_path)}")
        print(f"  Missing eval result path: {eval_result_path} {os.path.exists(eval_result_path)}")
        continue

# --- Build data for plotting ---
gaps = []          # x-axis: train - eval
sample_counts = [] # y-axis: number of samples
labels = []
for ds in dataset_num_samples:
    if ds in train_results and ds in eval_results:
        gaps.append(train_results[ds] - eval_results[ds])
        sample_counts.append(dataset_num_samples[ds])
        labels.append(ds)

# Sort points by gap (optional for consistent labeling order)
sorted_idx = sorted(range(len(gaps)), key=lambda i: gaps[i])
gaps = [gaps[i] for i in sorted_idx]
sample_counts = [sample_counts[i] for i in sorted_idx]
labels = [labels[i] for i in sorted_idx]

# Convert gap to percentage
gaps_pct = [g * 100.0 for g in gaps]

# Function to draw a subset plot given masks and output path
def draw_subset(mask, title, out_path, concise=False, smooth_curve=False, use_broken_axes=False, fig_width=10, fig_height=6):
    xs = [sample_counts[i] for i in range(len(sample_counts)) if mask[i]]
    ys = [gaps_pct[i]        for i in range(len(gaps_pct))     if mask[i]]
    lbs = [labels[i]         for i in range(len(labels))        if mask[i]]
    if len(xs) == 0:
        print(f"No points for: {title}. Skipping {out_path}")
        return

    # Basic stats and distinct counts for robustness
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    pos_x = xs_arr[xs_arr > 0]
    distinct_pos_x = np.unique(pos_x).size

    # Choose plotting backend: broken axes or normal axes
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_main = None
    bax = None

    # auto-detect breaks
    # x_segments = [(1e3-1, 2*1e4), (4*1e4, 5*1e4), (7*1e4, 9*1e4)]
    x_segments = [(8e2, 1e4*1.2), (4.67e4, 4.93e4), (7.78e4, 8e4*1.01)]   
    y_segments = [(2, 26), (51, 62)]
    # ensure at least one segment for each
    if not x_segments:
        xmin, xmax = (min([v for v in xs if v > 0]), max(xs)) if any(v > 0 for v in xs) else (min(xs), max(xs))
        x_segments = [(xmin, xmax)]
    if not y_segments:
        y_segments = [(min(ys), max(ys))]
    # Fix degenerate segments (zero-width/height) before creating broken axes
    def _fix_log_x_segs(segs):
        fixed = []
        for a, b in segs:
            lo = max(float(a), 1e-6)
            hi = max(float(b), 1e-6)
            if hi <= lo:
                pivot = max(lo, 1e-6)
                lo = pivot / 1.1
                hi = pivot * 1.1
            fixed.append((lo, hi))
        return fixed
    def _fix_lin_y_segs(segs):
        fixed = []
        for a, b in segs:
            a = float(a); b = float(b)
            if b <= a:
                pivot = 0.5 * (a + b)
                pad = max(0.5, 0.01 * max(1.0, abs(pivot)))
                a = pivot - pad
                b = pivot + pad
            fixed.append((a, b))
        return fixed
    x_segments = _fix_log_x_segs(x_segments)
    y_segments = _fix_lin_y_segs(y_segments)
    try:
        # Increase wspace to visually separate horizontal segments and avoid edge overlap illusion
        bax = brokenaxes(xlims=tuple(x_segments), ylims=tuple(y_segments), 
                         hspace=0.05, wspace=0.05,
                         d=0.006,              
                         tilt=55, 
                         despine=False)

        # set log scale on x segments
        for ax in np.ravel(bax.axs):
            # Avoid degenerate log transforms
            try:
                ax.set_xscale('log')
            except Exception as e:
                print(f"[brokenaxes] Log scale failed on sub-axes: {e}")
        scatter_artist = bax.scatter(xs, ys, c="#D68464", edgecolors='black', alpha=0.85)

    except Exception as e:
        print(f"[brokenaxes] Fallback to normal axes due to error: {e}")
        ax_main = fig.add_subplot(111)
        ax_main.scatter(xs, ys, c='tab:blue', edgecolors='black', alpha=0.85)

    import matplotlib.ticker as mticker
    from matplotlib.ticker import FixedLocator
    custom_ticks = {
        (4.7e4, 4.7e4): [47000, 47000],
        (7.9e4, 7.9e4*1.01): [79000, 79000],
    }
    for ax in np.ravel(bax.axs):
        lo, hi = ax.get_xlim()

        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis='x', which='minor', length=0)
        # ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
        # ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        # ax.tick_params(axis='x', which='minor', length=3)
        # ------------- 新增部分：为每段单独设置主刻度 -------------
        if 4.6e4 <= lo <= 5e4:
            ax.xaxis.set_major_locator(FixedLocator([4.8e4]))
        elif 7.6e4 <= lo <= 8.1e4:
            ax.xaxis.set_major_locator(FixedLocator([7.9e4]))
        # else:
        #     ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))

        # else:  # 第一段维持 log 刻度
        #     ax.xaxis.set_major_locator(
        #         mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        #     )
        
    bax.set_xlabel('Number of Training Samples', labelpad=20)
    bax.set_ylabel('Train Accuracy − Eval Accuracy (%)')
    # brokenaxes doesn't directly support set_title on the container in older versions

    # Save figure to path (create directory if needed)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved figure: {out_path}")
    except Exception as e:
        print(f"Failed to save figure {out_path}: {e}")
    finally:
        plt.close()

# Draw figure (wider width for readability)
draw_subset([True] * len(sample_counts), 'Train–Eval Gap', 'plots2/8_oracle_overfitting.png', concise=True, smooth_curve=True, use_broken_axes=True, fig_width=12, fig_height=8)
