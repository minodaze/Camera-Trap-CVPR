#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=bioclip2_best_accum_s
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --gpus-per-node=1         # One GPU per node
#SBATCH --cpus-per-task=8

USER_NAME="mino"
CONDA_ENV="ICICLE"

# Load your env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

DATA_ROOT="/fs/scratch/PAS2099/camera-trap-benchmark/dataset"
CONFIG_ROOT="/fs/ess/PAS2099/camera-trap-CVPR-configs"
# /fs/scratch/PAS2099/camera-trap-final/configs
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

    CONFIG_FILE="${CONFIG_ROOT}/${DATASET//\//_}/best_accum_lr${LEARNING_RATE}.yaml"

    mkdir -p "${CONFIG_ROOT}/${DATASET//\//_}"
    mkdir -p "/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/speciesnet_best_accum/${DATASET//\//_}"

    cat <<EOF > $CONFIG_FILE
module_name: best_accum
log_path: /fs/scratch/PAS2099/camera-trap-ECCV/ascend3/speciesnet_best_accum/${DATASET//\//_}

label_type: common

speciesnet_aliases:
  song thrush: ["song thrush"]
  domestic cat: ["domestic cat"]
  european rabbit: ["european rabbit"]
  common brushtail possum: ["common brushtail possum"]
  swamp harrier: ["northern harrier"]
  eurasian blackbird: ["common blackbird"]
  rat: ["australian swamp rat", "elegant rice rat", "desert kangaroo rat", "gambian rat", "polynesian rat", "rattus species", "broad-toothed rat", "fat sand rat", "chisel-toothed kangaroo rat", "malagasy giant jumping rat", "edward's rat", "huallaga spiny rat", "hoary bamboo rat", "bushy-tailed woodrat", "barbara brown's brush-tailed rat", "cuvier's spiny rat", "chinese bamboo rat", "desert woodrat", "eastern woodrat", "yellow-spotted brush-furred rat", "watson's climbing rat", "spiny rat family", "bush rat", "eastern red forest rat", "california kangaroo rat", "common rock rat", "white-throated woodrat", "swamp rat", "indomalayan bamboo rat", "petter's tuft-tailed rat", "oriental house rat", "webb's tuft-tailed rat", "southern plains woodrat", "ord's kangaroo rat", "white-tipped tuft-tailed rat", "big-eared swamp rat", "large spiny rat", "tome's spiny rat", "tanala tuft-tailed rat", "african wading rat", "charming thicket rat", "lesser cane rat", "forest giant pouched rat", "house rat", "giant kangaroo rat", "audebert's forest rat", "merriam's kangaroo rat", "mexican woodrat", "greater cane rat", "crested rat", "brown rat", "dusky-footed woodrat", "hispid cotton rat", "malaysian field rat", "woosnam's brush-furred rat", "long-tailed giant rat", "cayenne spiny rat", "common water rat", "rufous-nosed rat"]
  brown hare: ["european hare"]
  mouse: ["cotton deermouse", "california mouse", "garden dormouse", "tullberg's soft-furred mouse", "house mouse", "pygmy mouse", "long-tailed field mouse", "edible dormouse", "congo forest mouse", "northern grasshopper mouse", "palawan pencil-tailed tree mouse", "north american deermouse", "sandy inland mouse", "striped field mouse", "hispid pocket mouse", "white-footed mouse", "western harvest mouse", "meadow jumping mouse", "spinifex hopping mouse", "woodland jumping mouse", "bristly mouse"]
  european hedgehog: ["western european hedgehog"]

common_config:
  model: speciesnet
  train_data_config_path: /fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFD_DCAMF06/30/train.json
  eval_data_config_path: /fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFD_DCAMF06/30/test.json

  train_batch_size: 128
  eval_batch_size: 512

  optimizer_name: AdamW
  optimizer_params:
    lr: 0.000025
    weight_decay: 0.0001

  scheduler: null
  scheduler_params: null
  chop_head: false

pretrain_config:
  pretrain: false
ood_config:
  method: all
al_config:
  method: all
cl_config:
  method: accumulative-scratch
  epochs: 30
  loss_type: ce

EOF

    echo "Running pipeline for ${DATASET} with LR=${LEARNING_RATE}"
    python run_pipeline.py --c $CONFIG_FILE --wandb --eval_per_epoch --save_best_model --pretrained_weights speciesnet --lora_bottleneck 8 --loss_type bsm

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