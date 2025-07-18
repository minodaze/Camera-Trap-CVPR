#!/bin/bash

# Multi-GPU Training Launcher Script for ICICLE Benchmark
# This script launches training on 4 GPUs with different datasets

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BRIGHT_GREEN='\033[1;32m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 ICICLE Benchmark Multi-GPU Training Launcher${NC}"
echo "=================================================="

# Check if Python script exists
if [ ! -f "multi_gpu_training.py" ]; then
    echo -e "${RED}Error: multi_gpu_training.py not found in current directory${NC}"
    exit 1
fi

# ===================================================================
# MANUALLY EDIT THESE 4 DATASET NAMES (format: nz/nz_EFH_HCAMB10)
# The script will automatically convert / to _ 
# ===================================================================
DATASET_INPUT_1="orinoquia/orinoquia_N25"
DATASET_INPUT_2="na/na_archbold_FL-32"
DATASET_INPUT_3="MAD/MAD_A05"
DATASET_INPUT_4="nz/nz_EFH_HCAMB10"
# ===================================================================

# ===================================================================
# MANUALLY EDIT CONDA ENVIRONMENT NAME
# ===================================================================
CONDA_ENV="icicle"
# ===================================================================

# Convert dataset names (replace / with _)
DATASET1=$(echo "$DATASET_INPUT_1" | sed 's/\//_/g')
DATASET2=$(echo "$DATASET_INPUT_2" | sed 's/\//_/g')
DATASET3=$(echo "$DATASET_INPUT_3" | sed 's/\//_/g')
DATASET4=$(echo "$DATASET_INPUT_4" | sed 's/\//_/g')

echo -e "${GREEN}📝 Using manually configured datasets${NC}"
echo -e "  Input format → Converted format"
echo -e "  $DATASET_INPUT_1 → $DATASET1"
echo -e "  $DATASET_INPUT_2 → $DATASET2"
echo -e "  $DATASET_INPUT_3 → $DATASET3"
echo -e "  $DATASET_INPUT_4 → $DATASET4"

echo ""
echo -e "${BLUE}📋 Final Training Configuration:${NC}"
echo "================================="
echo "  GPU 0: $DATASET1"
echo "  GPU 1: $DATASET2"
echo "  GPU 2: $DATASET3"
echo "  GPU 3: $DATASET4"
echo ""

# Validate that datasets exist
CONFIG_ROOT="/fs/scratch/PAS2099/camera-trap-final/configs"
WORKSPACE_ROOT="/fs/ess/PAS2099/sooyoung/ICICLE-Benchmark"

echo -e "${YELLOW}Validating dataset directories...${NC}"
for dataset in $DATASET1 $DATASET2 $DATASET3 $DATASET4; do
    if [ ! -d "$CONFIG_ROOT/$dataset" ]; then
        echo -e "${RED}Error: Dataset directory not found: $CONFIG_ROOT/$dataset${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} $dataset"
done

echo ""
echo -e "${YELLOW}Each GPU will run the following training sequence:${NC}"
echo "  1. LoRA BSM (bottleneck=8): --c <dataset>_accu_bsm.yaml --lora_bottleneck 8"
echo "  2. LoRA CE (bottleneck=8):  --c <dataset>_accu_ce.yaml --lora_bottleneck 8"
echo "  3. Full CE:                 --c <dataset>_accu_ce.yaml --full"
echo "  4. Full BSM:                --c <dataset>_accu_bsm.yaml --full"
echo ""

# Ask for confirmation
read -p "Do you want to start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo -e "${MAGENTA}Starting Multi-GPU Training...${NC}"
echo "Log file: multi_gpu_training.log"
echo ""

# Run the Python script
python multi_gpu_training.py \
    --datasets $DATASET1 $DATASET2 $DATASET3 $DATASET4 \
    --config-root "$CONFIG_ROOT" \
    --workspace "$WORKSPACE_ROOT" \
    --conda-env "$CONDA_ENV"

echo ""
echo -e "${GREEN}🎉 Training launcher completed!${NC}"
