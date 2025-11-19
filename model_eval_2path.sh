#!/usr/bin/env bash

CONFIG_DIR="/fs/scratch/PAS2099/Lemeng/icicle/camera-trap-final/finalConfig"
PREF_B="/fs/ess/PAS2099/camera-trap-CVPR-logs/accum_80/best_accum"   # preferred
PREF_A="/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_accum"

# helper: resolve the directory that directly contains ckp_*_best_model.pth
resolve_model_dir() {
  local root="$1" ds="$2"
  local base="${root}/${ds}"
  local nested="${base}/bioclip2/lora_8_text_head/all/log"

  # 1) known nested path (accum_80 layout)
  if [[ -d "$nested" ]] && compgen -G "$nested/ckp_*_best_model.pth" > /dev/null; then
    printf '%s' "$nested"; return 0
  fi

  # 2) dataset root (sooyoung layout)
  if [[ -d "$base" ]] && compgen -G "$base/ckp_*_best_model.pth" > /dev/null; then
    printf '%s' "$base"; return 0
  fi

  # 3) last resort: search a few levels for any checkpoint, pick the dir of the "largest" (natural sort)
  if [[ -d "$base" ]]; then
    local any_ckp
    any_ckp="$(find "$base" -maxdepth 6 -type f -name 'ckp_*_best_model.pth' 2>/dev/null | sort -V | tail -n 1)"
    if [[ -n "$any_ckp" ]]; then
      dirname "$any_ckp"; return 0
    fi
  fi

  return 1
}

# datasets (one per line)
read -r -d '' DATASETS <<'EOF'
ENO_ENO_D06
serengeti_serengeti_N04
serengeti_serengeti_H11
APN_APN_K051
nz_nz_EFH_HCAME05
na_na_lebec_CA-21
MAD_MAD_A04
MTZ_MTZ_F04
APN_APN_13U
nz_nz_EFD_DCAMG03
nz_nz_PS1_CAM6213
nz_nz_EFH_HCAME09
MAD_MAD_D04
wellington_wellington_031c
KAR_KAR_A01
na_na_lebec_CA-18
serengeti_serengeti_Q09
APN_APN_U43B
APN_APN_K082
APN_APN_N1
nz_nz_EFH_HCAME08
serengeti_serengeti_L06
nz_nz_EFH_HCAMF01
caltech_caltech_46
serengeti_serengeti_S11
serengeti_serengeti_O13
PLN_PLN_B04
MTZ_MTZ_E05
nz_nz_EFD_DCAMH07
caltech_caltech_70
nz_nz_EFH_HCAMB05
KAR_KAR_B03
serengeti_serengeti_D02
MAD_MAD_B03
nz_nz_EFD_DCAMF06
caltech_caltech_88
na_na_lebec_CA-19
APN_APN_U23A
na_na_lebec_CA-05
nz_nz_EFH_HCAMI01
KGA_KGA_KHOGA04
ENO_ENO_C02
ENO_ENO_C04
MAD_MAD_C07
serengeti_serengeti_E05
serengeti_serengeti_V10
na_na_lebec_CA-31
serengeti_serengeti_F08
MAD_MAD_B06
nz_nz_EFH_HCAMB01
KGA_KGA_KHOLA03
nz_nz_EFH_HCAMD08
nz_nz_EFH_HCAMC03
serengeti_serengeti_L10
serengeti_serengeti_D09
idaho_idaho_122
serengeti_serengeti_Q11
MAD_MAD_H08
CDB_CDB_A05
serengeti_serengeti_E12
nz_nz_EFH_HCAMC02
serengeti_serengeti_T10
serengeti_serengeti_H03
nz_nz_PS1_CAM8008
na_na_lebec_CA-37
serengeti_serengeti_R10
nz_nz_EFD_DCAMH01
nz_nz_EFH_HCAMG13
serengeti_serengeti_K11
APN_APN_WM
nz_nz_PS1_CAM7312
ENO_ENO_E06
serengeti_serengeti_Q10
serengeti_serengeti_H08
APN_APN_TB17
serengeti_serengeti_Q07
caltech_caltech_38
MTZ_MTZ_D06
nz_nz_EFD_DCAMD10
MTZ_MTZ_D03
EOF

# run per dataset
while IFS= read -r ds; do
  [[ -z "$ds" ]] && continue
  cfg="${CONFIG_DIR}/${ds}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[skip] $ds — missing config: $cfg"
    continue
  fi

  model_dir=""
  # prefer accum_80
  if model_dir="$(resolve_model_dir "$PREF_B" "$ds")"; then
    :
  elif model_dir="$(resolve_model_dir "$PREF_A" "$ds")"; then
    :
  else
    echo "[skip] $ds — no checkpoint files found under:"
    echo "       - ${PREF_B}/${ds}"
    echo "       - ${PREF_A}/${ds}"
    continue
  fi

  echo "[run] $ds"
  echo "      cfg       : $cfg"
  echo "      model_dir : $model_dir"
  conda run -n icicle_env python run_pipeline.py \
    --c "$cfg" \
    --wandb --eval_per_epoch --test_per_epoch --save_best_model \
    --lora_bottleneck 8 --eval_only \
    --model_dir "$model_dir" \
    --accu_eval
done <<< "$DATASETS"
