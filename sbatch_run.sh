#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=bioclip2_upper_bound
#SBATCH --output=logs/bioclip2_%j.out
#SBATCH --error=logs/bioclip2_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1                 # Request 4 nodes
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --gpus-per-node=1         # One GPU per node
#SBATCH --cpus-per-task=12

USER_NAME="jeonso193"
CONDA_ENV="icicle"

# Load your env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

DATA_ROOT="/fs/scratch/PAS2099/camera-trap-benchmark"
CONFIG_ROOT="/fs/scratch/PAS2099/${USER_NAME}/icicle/configs/generated_common"
# CSV_PATH="/fs/ess/PAS2099/${USER_NAME}/Documents/icicle/ICICLE-Benchmark/balanced_accuracy_common.csv"

mkdir -p $CONFIG_ROOT
# mkdir -p $(dirname "$CSV_PATH")

BIG_FOLDERS=(
    "idaho/idaho_51"
    "idaho/idaho_30"
    "nz/nz_FGA_1"
    "nz/nz_AHO_H2.2.1"
)


for DATASET in "${BIG_FOLDERS[@]}"; do
    echo "=== Processing ${DATASET} ==="
    TRAIN_JSON="${DATA_ROOT}/${DATASET}/30/train.json"
    TEST_JSON="${DATA_ROOT}/${DATASET}/30/test.json"
    ALL_JSON="${DATA_ROOT}/${DATASET}/30/train-all.json"
    # === Generate outer timestamp ===
    PARENT_TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
    # === Extract class names ===
#     CLASS_NAMES=$(python -c "
# import json
# with open('${ALL_JSON}') as f:
#     data = json.load(f)
# common = sorted(set(item['common'] for item in data['ckp_-1']))
# print('\n'.join(['  - ' + s for s in common]))
# ")

    CONFIG_FILE="${CONFIG_ROOT}/${DATASET//\//_}_upper_bound.yaml"

    cat <<EOF > $CONFIG_FILE
module_name: upper_bound
log_path: /fs/scratch/PAS2099/${USER_NAME}/icicle/log_auto/pipeline/${DATASET//\//_}/upper_bound/${PARENT_TIMESTAMP}/

common_config:
  model: bioclip2
  train_data_config_path: ${TRAIN_JSON}
  eval_data_config_path: ${TEST_JSON}
  all_data_config_path: ${ALL_JSON}
  train_batch_size: 32
  eval_batch_size: 512
  optimizer_name: AdamW
  optimizer_params:
    lr: 0.000025
    weight_decay: 0.0001
  chop_head: false
  scheduler: CosineAnnealingLR
  scheduler_params:
    T_max: 30
    eta_min: 0.0000025

pretrain_config:
  pretrain: true
  pretrain_data_config_path: ${ALL_JSON}
  epochs: 60
  loss_type: ce

ood_config:
  method: none

al_config:
  method: none

cl_config:
  method: none
EOF

    echo "Running pipeline for ${DATASET}"
    python run_pipeline.py --c $CONFIG_FILE --full --wandb

#     # === Robust log path discovery ===
#     BASE_LOG_DIR="/fs/scratch/PAS2099/${USER_NAME}/icicle/log_auto/pipeline/${DATASET//\//_}/zs_common/${PARENT_TIMESTAMP}/"

#     echo "Searching for nested logs in: ${BASE_LOG_DIR}"
#     echo "Contents:"
#     ls -lah ${BASE_LOG_DIR}

#     # Find the latest nested bioclip2/full_text_head/*/
#     SUB_TS=$(ls -td ${BASE_LOG_DIR}bioclip2/full_text_head/*/ | head -n1)

#     LOG_PATH="${SUB_TS}log/log.txt"

#     echo "Latest log path: ${LOG_PATH}"

#     if [ ! -f "$LOG_PATH" ]; then
#       echo "Log file does not exist: ${LOG_PATH}"
#       continue
#     fi

#     echo "Parsing and appending for ${DATASET}"
#     python parse_and_append.py --dataset "${DATASET}" --log_path "${LOG_PATH}" --csv_path "${CSV_PATH}"

#   done
done