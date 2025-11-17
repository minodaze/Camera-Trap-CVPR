import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open('/users/PAS2099/mino/ICICLE/plot/oracle.txt', 'r') as f:
    datasets = f.read().splitlines()
# /fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_ascend/na_na_archbold_FL-09/bioclip2/full_text_head/all/log/avg_ub_all_log.json
def read_bsm(dataset: str) -> str:
    best_json_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_ascend/{dataset}/bioclip2/full_text_head/all/log/final_training_summary.json"
    all_log_json = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_ascend/{dataset}/bioclip2/full_text_head/all/log/avg_ub_all_log.json"
    if not os.path.exists(best_json_path):
        best_json_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_cardinal/{dataset}/bioclip2/full_text_head/all/log/final_training_summary.json"
        all_log_json = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_cardinal/{dataset}/bioclip2/full_text_head/all/log/avg_ub_all_log.json"
        if not os.path.exists(best_json_path):
            best_json_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_multi_pitzer/{dataset}/bioclip2/full_text_head/all/log/final_training_summary.json"
            all_log_json = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_multi_pitzer/{dataset}/bioclip2/full_text_head/all/log/avg_ub_all_log.json"
    if not os.path.exists(best_json_path):
        raise FileNotFoundError(f"BSM JSON not found for dataset {dataset}")

    return best_json_path, all_log_json

# /fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle_test_per_epoch_ascend/APN_APN_U64C/bioclip2/lora_8_text_head/all/log/final_training_summary.json
def read_lora(dataset: str) -> str:
    best_json_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle_test_per_epoch_ascend/{dataset}/bioclip2/lora_8_text_head/all/log/final_training_summary.json"
    all_log_json = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle_test_per_epoch_ascend/{dataset}/bioclip2/lora_8_text_head/all/log/avg_ub_all_log.json"
    if not os.path.exists(best_json_path):
        best_json_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle_test_per_epoch_cardinal/{dataset}/bioclip2/lora_8_text_head/all/log/final_training_summary.json"
        all_log_json = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle_test_per_epoch_cardinal/{dataset}/bioclip2/lora_8_text_head/all/log/avg_ub_all_log.json"
        if not os.path.exists(best_json_path):
            best_json_path = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle_test_per_epoch_multi_pitzer/{dataset}/bioclip2/lora_8_text_head/all/log/final_training_summary.json"
            all_log_json = f"/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/lora_oracle_test_per_epoch_multi_pitzer/{dataset}/bioclip2/lora_8_text_head/all/log/avg_ub_all_log.json"
    if not os.path.exists(best_json_path):
        raise FileNotFoundError(f"LORA JSON not found for dataset {dataset}")

    return best_json_path, all_log_json

def extract_curve(all_log: dict):
    epochs, vals = [], []
    # Expect keys like "epoch_0", "epoch_1" ... each containing nested metrics
    for k, v in all_log.items():
        if not isinstance(k, str) or not k.startswith("epoch_"):
            continue
        try:
            ei = int(k.split("_")[1])
        except Exception:
            continue
        ba = None
        if isinstance(v, dict):
            ba = v.get("test_avg", {}).get("avg_balanced_acc", None)
        if ba is None:
            continue
        epochs.append(ei)
        vals.append(float(ba))
    if not epochs:
        return [], []
    order = np.argsort(epochs)
    epochs = [epochs[i] for i in order]
    vals = [vals[i] for i in order]
    return epochs, vals

def match_best_epoch(epochs, vals, target, tol=1e-9):
    if not epochs:
        return None
    diffs = [abs(v - target) for v in vals]
    i = int(np.argmin(diffs))
    if diffs[i] > tol:
        # Fallback to the epoch with the maximum value if exact match not found
        i = int(np.argmax(vals))
    return epochs[i], vals[i], i

out_dir = "/users/PAS2099/mino/ICICLE/plots2/peft_imbalance_loss_bal_acc_curves"
os.makedirs(out_dir, exist_ok=True)

summary_rows = []
for dataset in datasets:
    try:
        bsm_best_json_path, bsm_all_log_json = read_bsm(dataset)
        lora_best_json_path, lora_all_log_json = read_lora(dataset)
    except FileNotFoundError as e:
        print(e)
        continue

    with open(bsm_best_json_path, 'r') as f:
        bsm_best_data = json.load(f)
    with open(bsm_all_log_json, 'r') as f:
        bsm_all_log_data = json.load(f)
    with open(lora_best_json_path, 'r') as f:
        lora_best_data = json.load(f)
    with open(lora_all_log_json, 'r') as f:
        lora_all_log_data = json.load(f)

    bsm_best_avg_ba = float(bsm_best_data['averages']['balanced_accuracy'])
    lora_best_avg_ba = float(lora_best_data['averages']['balanced_accuracy'])

    bsm_epochs, bsm_vals = extract_curve(bsm_all_log_data)
    lora_epochs, lora_vals = extract_curve(lora_all_log_data)

    bsm_match = match_best_epoch(bsm_epochs, bsm_vals, bsm_best_avg_ba)
    lora_match = match_best_epoch(lora_epochs, lora_vals, lora_best_avg_ba)

    if bsm_match is None and lora_match is None:
        print(f"No epochs found for {dataset}")
        continue

    plt.figure(figsize=(8, 5))
    if bsm_match:
        bsm_best_epoch, bsm_best_val, bsm_idx = bsm_match
        plt.plot(bsm_epochs[:bsm_idx+1], np.array(bsm_vals[:bsm_idx+1]) * 100.0,
                 marker='o', label='BSM', color='#2a9d8f')
        plt.scatter([bsm_best_epoch], [bsm_best_val * 100.0],
                    color='#2a9d8f', edgecolors='black', zorder=3)
    if lora_match:
        lora_best_epoch, lora_best_val, lora_idx = lora_match
        plt.plot(lora_epochs[:lora_idx+1], np.array(lora_vals[:lora_idx+1]) * 100.0,
                 marker='s', label='LoRA', color='#e76f51')
        plt.scatter([lora_best_epoch], [lora_best_val * 100.0],
                    color='#e76f51', edgecolors='black', zorder=3)

    # Ensure x-axis ticks are integer epochs
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(dataset)
    plt.xlabel("Epoch")
    plt.ylabel("Test Balanced Accuracy (%)")
    plt.grid(True, ls='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{dataset.replace('/', '_')}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    if bsm_match:
        summary_rows.append((dataset, 'BSM', bsm_match[0], bsm_match[1]))
    if lora_match:
        summary_rows.append((dataset, 'LoRA', lora_match[0], lora_match[1]))

for ds, method, ep, val in summary_rows:
    print(f"{ds:40s}  {method:4s} best_epoch={ep:2d}  best_avg_bal_acc={val:.4f}")