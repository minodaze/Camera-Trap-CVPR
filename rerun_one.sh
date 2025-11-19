#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=when2adapt-na_na_lebec_CA-18
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=output/na_na_lebec_CA-18_%j.out
#SBATCH --error=output/na_na_lebec_CA-18_%j.err

set -euo pipefail

# --- user env (edit if needed) ---
module load cuda/12.1 || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate icicle_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_LAUNCH_BLOCKING=0

mkdir -p output

# --- inputs ---
PY_SCRIPT="new_problem_setting_saving_feature_new_threshold.py"
DATASET="na_na_lebec_CA-18"   # single dataset to rerun

echo "[Job ${SLURM_JOB_ID}] Rerunning dataset: ${DATASET}"
python "${PY_SCRIPT}" --datasets "${DATASET}"
