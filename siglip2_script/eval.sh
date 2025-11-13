#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=Siglip2_zs_eval
#SBATCH --output=/fs/ess/PAS2099/sooyoung/Camera-Trap-CVPR/siglip2_script/output/siglip2_%j.out
#SBATCH --error=/fs/ess/PAS2099/sooyoung/Camera-Trap-CVPR/siglip2_script/error/siglip2_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1                 # Request 4 nodes
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --gpus-per-node=1         # One GPU per node
#SBATCH --cpus-per-task=12

CONFIG_DIR="/fs/ess/PAS2099/camera-trap-CVPR-configs"
CODE_DIR="/fs/ess/PAS2099/sooyoung/Camera-Trap-CVPR"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate icicle

# Change to code directory
cd "$CODE_DIR"

for YAML in "$@"; do
    echo "Running pipeline with config: $YAML"
    python run_pipeline.py --c "$YAML" --eval_per_epoch --full --pretrained_weights siglip2 --text_template siglip2 --wandb
done