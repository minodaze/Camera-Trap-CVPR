#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=bioclip2_upper_bound
#SBATCH --output=logs/bioclip2_%j.out
#SBATCH --error=logs/bioclip2_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1                 # Request 4 nodes
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --gpus-per-node=1         # One GPU per node
#SBATCH --cpus-per-task=12

USER_NAME="mino"
CONDA_ENV="ICICLE"

interpolation_alpha=(
    0.0
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
    1.0
)

# Load your env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

DATA_ROOT="/fs/scratch/PAS2099/camera-trap-benchmark/dataset"
CONFIG_ROOT="/fs/scratch/PAS2099/${USER_NAME}/ICICLE/configs/generated_common"
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
MODEL_DIR="$3"

echo "Processing ${#BIG_FOLDERS[@]} datasets: ${BIG_FOLDERS[*]}"
echo "Using learning rate: ${LEARNING_RATE}"
echo "Using model directory: ${MODEL_DIR}"

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

    CONFIG_FILE="${CONFIG_ROOT}/${DATASET//\//_}_eval_interpolation_lr${LEARNING_RATE}.yaml"

    cat <<EOF > $CONFIG_FILE
module_name: eval_full_interpolation
log_path: /fs/scratch/PAS2099/camera-trap-final/round2_eval_logs/${DATASET//\//_}/eval_full_lora_accum_interpolation/

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
  pretrain: false
ood_config:
  method: none
al_config:
  method: none
cl_config:
  method: none
EOF

    # Run pipeline for each interpolation alpha value
    for alpha in "${interpolation_alpha[@]}"; do
        echo "Running pipeline for ${DATASET} with LR=${LEARNING_RATE} and alpha=${alpha}"
        python run_pipeline.py --c $CONFIG_FILE --eval_only --model_dir "${MODEL_DIR}" --pretrained_weights bioclip2 --lora_bottleneck 8 --merge_factor 0.8
    done
done