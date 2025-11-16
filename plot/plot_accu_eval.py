import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open('/users/PAS2099/mino/ICICLE/uselist/eval_dataset.txt', 'r') as f:
    datasets = f.read().splitlines()

df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Best Accum Analysis.csv')
csv_output_path = '/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - Best Accum Analysis with Accu Eval.csv'


df['accu eval stop 1/3'] = 0.0
df['accu eval stop 2/3'] = 0.0

def plot_accu_eval(ds):
    json_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum_accu_eval_all/{ds}/bioclip2/lora_8_text_head/all/log/eval_accu_eval_only_summary.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter and sort checkpoint keys (ignore non-ckp keys like 'averages')
    ckp_keys = sorted([k for k in data.keys() if k.startswith('ckp_')], key=lambda s: int(s.split('_')[-1]))
    length = len(ckp_keys)
    if length == 0:
        print(f"No checkpoint entries found for {ds}")
        return

    def get_bal_acc(outer, inner):
        try:
            return float(data[outer][inner]['balanced_accuracy'])
        except Exception:
            return None

    # Main line: diagonal metrics data[ckp_i][ckp_i]
    main_line_results = []
    for k in ckp_keys:
        v = get_bal_acc(k, k)
        main_line_results.append(v if v is not None else float('nan'))

    # Stop indices (1-based): L//3 + 1 and 2L//3 + 1
    stop1_idx = max(1, min(length, length // 3 + 1))
    stop2_idx = max(1, min(length, (2 * length) // 3 + 1))

    def build_early_stop_line(stop_idx):
        stop_key = f"ckp_{stop_idx}"
        line = []
        for j, kj in enumerate(ckp_keys, start=1):
            if j <= stop_idx:
                line.append(main_line_results[j - 1])
            else:
                v = get_bal_acc(stop_key, kj)
                line.append(v if v is not None else float('nan'))
        return line

    stop_1_over_3_results = build_early_stop_line(stop1_idx)
    stop_2_over_3_results = build_early_stop_line(stop2_idx)

    # Plotting
    out_dir = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/accu_eval_best_accum_fig/{ds}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accu_eval_best_accum_fig.png")

    x = list(range(1, length + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(x, main_line_results, marker='o', label='100% Training', color='lightgreen')
    plt.plot(x, stop_1_over_3_results, marker='o', linestyle='--', label=f'Stop @ 1/3', color='orange')
    plt.plot(x, stop_2_over_3_results, marker='o', linestyle='--', label=f'Stop @ 2/3', color='lightblue')
    plt.title(f"Balanced Accuracy vs Checkpoint — {ds}")
    plt.xlabel("Checkpoint index")
    plt.ylabel("Balanced accuracy")
    plt.xticks(x, [k.split('_')[-1] for k in ckp_keys])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    df.loc[df['dataset'] == ds, 'accu eval stop 1/3'] = np.mean([v for v in stop_1_over_3_results if not pd.isna(v)])
    df.loc[df['dataset'] == ds, 'accu eval stop 2/3'] = np.mean([v for v in stop_2_over_3_results if not pd.isna(v)])


for dataset in datasets:
    try:
        plot_accu_eval(dataset)
    except Exception as e:
        print(f"Error processing {dataset}: {e}")

df.to_csv(csv_output_path, index=False)