#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=bioclip2_zs_mcm_subset
#SBATCH --output=logs/bioclip2_%j.out
#SBATCH --error=logs/bioclip2_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16

set -euo pipefail

# --- Conda env ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate icicle_env

CONFIG_ROOT="/fs/ess/PAS2099/camera-trap-CVPR-configs"

# List of dataset IDs (matching CONFIG_ROOT/<ID>/zs_lr0.000025.yaml)
DATASET_IDS=(
  # "nz_nz_EFH_HCAMB05"
  # "serengeti_serengeti_L10"
  # "serengeti_serengeti_K11"
  # "serengeti_serengeti_S11"
  # "serengeti_serengeti_F08"
  # "serengeti_serengeti_H11"
  # "nz_nz_EFH_HCAMC02"
  # "serengeti_serengeti_Q09"
  # "nz_nz_PS1_CAM8008"
  # "na_na_lebec_CA-21"
  # "ENO_ENO_C02"
  # "nz_nz_EFH_HCAMF01"
  # "serengeti_serengeti_E12"
  # "nz_nz_EFH_HCAMD08"
  # "serengeti_serengeti_E05"
  # "nz_nz_EFH_HCAMI01"
  # "wellington_wellington_031c"
  # "nz_nz_EFH_HCAMG13"
  # "serengeti_serengeti_V10"
  # "nz_nz_EFD_DCAMH07"
  # "serengeti_serengeti_N04"
  # "serengeti_serengeti_Q10"
  "nz_nz_EFH_HCAMC03"
  "serengeti_serengeti_L06"
  "nz_nz_PS1_CAM6213"
  "serengeti_serengeti_Q07"
  "nz_nz_PS1_CAM7312"
  "serengeti_serengeti_H08"
  "APN_APN_N1"
  "serengeti_serengeti_H03"
  "serengeti_serengeti_D02"
  "ENO_ENO_C04"
  "serengeti_serengeti_D09"
  "MTZ_MTZ_E05"
  "KAR_KAR_B03"
  "PLN_PLN_B04"
  "MAD_MAD_A04"
  "ENO_ENO_E06"
  "na_na_lebec_CA-37"
  "KGA_KGA_KHOGA04"
  # "APN_APN_TB17"
  # "nz_nz_EFD_DCAMF06"
  # "serengeti_serengeti_Q11"
  # "MTZ_MTZ_D03"
  # "CDB_CDB_A05"
  # "MAD_MAD_H08"
  # "APN_APN_WM"
  # "MTZ_MTZ_F04"
  # "MAD_MAD_B03"
  # "na_na_lebec_CA-31"
  # "nz_nz_EFH_HCAMB01"
  # "APN_APN_13U"
  # "MTZ_MTZ_D06"
  # "nz_nz_EFH_HCAME08"
  # "nz_nz_EFD_DCAMD10"
  # "nz_nz_EFH_HCAME05"
  # "na_na_lebec_CA-19"
  # "APN_APN_K051"
  # "KAR_KAR_A01"
  # "caltech_caltech_88"
  # "serengeti_serengeti_R10"
  # "na_na_lebec_CA-05"
  # "MAD_MAD_D04"
  # "APN_APN_K082"
  "APN_APN_U23A"
  "APN_APN_U43B"
  "caltech_caltech_38"
  "caltech_caltech_46"
  "caltech_caltech_70"
  "ENO_ENO_D06"
  "idaho_idaho_122"
  "KGA_KGA_KHOLA03"
  "MAD_MAD_B06"
  "MAD_MAD_C07"
  "na_na_lebec_CA-18"
  "nz_nz_EFD_DCAMG03"
  "nz_nz_EFD_DCAMH01"
  "nz_nz_EFH_HCAME09"
  "serengeti_serengeti_O13"
  "serengeti_serengeti_T10"
)

NUM=${#DATASET_IDS[@]}
HALF=$(( NUM / 2 ))

echo "Total datasets: ${NUM}, first half: 0..$((HALF-1)), second half: ${HALF}..$((NUM-1))"

# --- First half on GPU 0 ---
(
  export CUDA_VISIBLE_DEVICES=0
  echo "=== GPU 0 handling indices 0 to $((HALF-1)) ==="
  for ((i=0; i<HALF; i++)); do
    ID="${DATASET_IDS[$i]}"

    BIG="${ID%%_*}"
    SMALL="${ID#*_}"
    DATASET="${BIG}/${SMALL}"

    CONFIG_FILE="${CONFIG_ROOT}/${ID}/zs_lr0.000025.yaml"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
      echo "GPU0 WARNING: Config not found: ${CONFIG_FILE} (skipping ${ID})"
      continue
    fi

    echo "GPU0: Running pipeline for ${DATASET} using ${CONFIG_FILE}"
    python run_pipeline.py --c "${CONFIG_FILE}" --full
  done
) &

# --- Second half on GPU 1 ---
(
  export CUDA_VISIBLE_DEVICES=1
  echo "=== GPU 1 handling indices ${HALF} to $((NUM-1)) ==="
  for ((i=HALF; i<NUM; i++)); do
    ID="${DATASET_IDS[$i]}"

    BIG="${ID%%_*}"
    SMALL="${ID#*_}"
    DATASET="${BIG}/${SMALL}"

    CONFIG_FILE="${CONFIG_ROOT}/${ID}/zs_lr0.000025.yaml"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
      echo "GPU1 WARNING: Config not found: ${CONFIG_FILE} (skipping ${ID})"
      continue
    fi

    echo "GPU1: Running pipeline for ${DATASET} using ${CONFIG_FILE}"
    python run_pipeline.py --c "${CONFIG_FILE}" --full
  done
) &

wait
echo "All requested datasets processed on both GPUs."
