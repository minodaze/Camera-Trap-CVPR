#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=bioclip2_upper_bound
#SBATCH --output=logs/bioclip2_%j.out
#SBATCH --error=logs/bioclip2_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --gpus-per-node=1         # One GPU per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # Request 100GB total memory per node

USER_NAME="mino"
CONDA_ENV="ICICLE"

# Load your env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

DATA_ROOT="/fs/scratch/PAS2099/camera-trap-benchmark/dataset"
CONFIG_ROOT="/fs/ess/PAS2099/camera-trap-CVPR-configs"
# CSV_PATH="/fs/ess/PAS2099/${USER_NAME}/Documents/ICICLE/ICICLE-Benchmark/balanced_accuracy_common.csv"

mkdir -p $CONFIG_ROOT
# mkdir -p $(dirname "$CSV_PATH")

# Get datasets and learning rate from command line arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch sbatch_run_1.sh 'dataset1 dataset2 dataset3' learning_rate"
    exit 1
fi

# Parse datasets from the first argument (space-separated string)
IFS=' ' read -ra BIG_FOLDERS <<< "$1"
# Get learning rate from the second argument
LEARNING_RATE="$2"

echo "Processing ${#BIG_FOLDERS[@]} datasets: ${BIG_FOLDERS[*]}"
echo "Using learning rate: ${LEARNING_RATE}"


for DATASET in "${BIG_FOLDERS[@]}"; do
    echo "=== Processing ${DATASET} ==="
    TRAIN_JSON="${DATA_ROOT}/${DATASET}/30/train.json"
    TEST_JSON="${DATA_ROOT}/${DATASET}/30/test.json"
    ALL_JSON="${DATA_ROOT}/${DATASET}/30/train-all.json"
    # === Generate deterministic timestamp based on dataset and learning rate ===
    # This ensures identical runs use the same directory, improving reproducibility
    HASH_INPUT="${DATASET}_${LEARNING_RATE}"
    PARENT_TIMESTAMP=$(echo -n "$HASH_INPUT" | sha256sum | cut -c1-16)
    PARENT_TIMESTAMP="$(date +%Y-%m-%d-%H)-$(echo $PARENT_TIMESTAMP | cut -c1-2)-$(echo $PARENT_TIMESTAMP | cut -c3-4)"
    # === Extract class names ===
#     CLASS_NAMES=$(python -c "
# import json
# with open('${ALL_JSON}') as f:
#     data = json.load(f)
# common = sorted(set(item['common'] for item in data['ckp_-1']))
# print('\n'.join(['  - ' + s for s in common]))
# ")

    CONFIG_FILE="${CONFIG_ROOT}/${DATASET//\//_}/bsm_oracle_lr${LEARNING_RATE}.yaml"

    mkdir -p "${CONFIG_ROOT}/${DATASET//\//_}"
    mkdir -p "/fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_ascend/${DATASET//\//_}"

    cat <<EOF > $CONFIG_FILE
module_name: oracle
log_path: /fs/ess/PAS2099/camera-trap-CVPR-logs/oracle_20/bsm_oracle_test_per_epoch_ascend/${DATASET//\//_}
common_config:
  model: bioclip2
  train_data_config_path: ${TRAIN_JSON}
  eval_data_config_path: ${TEST_JSON}
  all_data_config_path: ${ALL_JSON}
  train_batch_size: 32
  eval_batch_size: 512
  optimizer_name: AdamW
  optimizer_params:
    lr: ${LEARNING_RATE}
    weight_decay: 0.0001
  chop_head: false
  scheduler: CosineAnnealingLR
  scheduler_params:
    T_max: 60
    eta_min: $(echo "${LEARNING_RATE} / 10" | bc -l)

pretrain_config:
  pretrain: true
  pretrain_data_config_path: ${ALL_JSON}
  epochs: 30
  loss_type: bsm
  
ood_config:
  method: all
al_config:
  method: all
cl_config:
  method: none
EOF

    echo "Running pipeline for ${DATASET} with LR=${LEARNING_RATE}"
    python run_pipeline.py --c $CONFIG_FILE --wandb --eval_per_epoch --test_per_epoch --save_best_model --pretrained_weights bioclip2 --full

#     # === Robust log path discovery ===
#     BASE_LOG_DIR="/fs/scratch/PAS2099/${USER_NAME}/ICICLE/log_auto/pipeline/${DATASET//\//_}/zs_common/"

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