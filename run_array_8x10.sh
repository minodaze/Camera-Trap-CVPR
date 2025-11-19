#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=when2adapt-6x5
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=output/array2_%a.out
#SBATCH --error=output/array2_%a.err

set -euo pipefail

# --- user env (edit if needed) ---
module load cuda/12.1 || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate icicle_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_LAUNCH_BLOCKING=0

# --- inputs ---
MASTER_LIST="dataset_left.txt"          # 80 lines, one dataset per line (slash-form)
PY_SCRIPT="new_problem_setting_saving_feature_new_threshold.py"            # path to your revised Python file
PER_TASK=5                           # 6 tasks * 5 datasets = 30

# Compute this task's slice [START_LINE, END_LINE]
START_LINE=$(( SLURM_ARRAY_TASK_ID * PER_TASK + 1 ))
END_LINE=$(( START_LINE + PER_TASK - 1 ))

mkdir -p output
CHUNK_FILE="chunk2_${SLURM_ARRAY_TASK_ID}.txt"

# Extract the slice; sed gracefully stops at EOF
sed -n "${START_LINE},${END_LINE}p" "${MASTER_LIST}" > "${CHUNK_FILE}"

if [[ ! -s "${CHUNK_FILE}" ]]; then
  echo "[Task ${SLURM_ARRAY_TASK_ID}] No datasets found in ${MASTER_LIST} for lines ${START_LINE}-${END_LINE}"
  exit 0
fi

# Turn the slice into a comma-separated list for --datasets
DATASETS_CSV=$(paste -sd, "${CHUNK_FILE}")
echo "[Task ${SLURM_ARRAY_TASK_ID}] Running datasets: ${DATASETS_CSV}"

# Run: the Python script writes per-dataset summaries under SUMMARY_ROOT/<ds_u>/summary.csv
python "${PY_SCRIPT}" --datasets "${DATASETS_CSV}"

# Cleanup (optional)
rm -f "${CHUNK_FILE}"
