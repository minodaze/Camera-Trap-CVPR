#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=ICICLE-Train
#SBATCH --time=05:00:00
#SBATCH --nodes=1                 # Request 4 nodes
#SBATCH --ntasks-per-node=1       # One task per node
#SBATCH --gpus-per-node=1         # One GPU per node
#SBATCH --cpus-per-task=24
#SBATCH --output=output/%j_%x.slurm.out

module load cuda/11.8.0
source /users/PAS2119/jeonso193/miniconda3/etc/profile.d/conda.sh
conda activate /fs/ess/PAS2099/sooyoung/envs/myenv

export WANDB_DIR=/fs/scratch/PAS2099/sooyoung/icicle-benchmark-logs

python run_pipeline.py --wandb --c /fs/ess/PAS2099/sooyoung/ICICLE-Benchmark/config/pipeline/nz_EFH_HCAMF02/percentage-1.yaml