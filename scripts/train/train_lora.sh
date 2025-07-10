#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=ICICLE-Train
#SBATCH --time=48:00:00
#SBATCH --nodes=1                 # Request 4 nodes
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --gpus-per-node=1         # One GPU per node
#SBATCH --cpus-per-task=24
#SBATCH --output=output/%j_%x.slurm.out

module load cuda/11.8.0
source /users/PAS2119/hou/miniconda3/etc/profile.d/conda.sh
conda activate /fs/scratch/PAS2119/hou/icicle_env

export WANDB_DIR=/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/wandb

python run_pipeline.py --wandb --c /users/PAS2119/hou/ICICLE/ICICLE-Benchmark/config/pipeline/ENO_C05_common-name/cdt/percentage-1.yaml --full
