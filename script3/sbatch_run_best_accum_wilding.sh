#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=wilding_accum_scratch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

CONDA_ENV="ICICLE"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

DATA_ROOT="/fs/scratch/PAS2099/camera-trap-benchmark/dataset"
LLM_JSON="config/LLM_description/species_descriptions.json"
DESC_CACHE="config/LLM_description/vlm_desc_cache.pt"

if [ $# -lt 2 ]; then
    echo "Usage: sbatch sbatch_run_best_accum_wilding.sh 'dataset1 dataset2' learning_rate"
    exit 1
fi

IFS=' ' read -ra BIG_FOLDERS <<< "$1"
LEARNING_RATE="$2"

mkdir -p logs

for DATASET in "${BIG_FOLDERS[@]}"; do
    echo "=== WildIng accumulative-scratch: ${DATASET} LR=${LEARNING_RATE} ==="

    TRAIN_JSON="${DATA_ROOT}/${DATASET}/30/train.json"
    TEST_JSON="${DATA_ROOT}/${DATASET}/30/test.json"
    OUTPUT_DIR="my_weights/wilding/${DATASET//\//_}_lr${LEARNING_RATE}"

    if [ ! -f "${TRAIN_JSON}" ]; then
        echo "ERROR: train.json not found at ${TRAIN_JSON}"
        continue
    fi
    if [ ! -f "${TEST_JSON}" ]; then
        echo "ERROR: test.json not found at ${TEST_JSON}"
        continue
    fi
    if [ ! -f "${DESC_CACHE}" ]; then
        echo "VLM desc cache not found. Running precompute_vlm_descriptions.py first..."
        python scripts/precompute_vlm_descriptions.py \
            --pretrained-weights bioclip2 \
            --output "${DESC_CACHE}"
    fi

    mkdir -p "${OUTPUT_DIR}"

    python train_wilding.py \
        --train-json        "${TRAIN_JSON}" \
        --test-json         "${TEST_JSON}"  \
        --llm-json          "${LLM_JSON}"   \
        --desc-cache        "${DESC_CACHE}" \
        --output-dir        "${OUTPUT_DIR}" \
        --pretrained-weights bioclip2       \
        --hidden-dim        793             \
        --alpha             0.5             \
        --tau               0.1             \
        --epochs            30              \
        --batch-size        128             \
        --lr                "${LEARNING_RATE}" \
        --momentum          0.80            \
        --num-workers       8               \
        --early-stop        5

    echo "Results saved to ${OUTPUT_DIR}/results.json"
done
