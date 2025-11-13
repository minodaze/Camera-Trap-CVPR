#!/bin/bash
CONFIG_DIR="/fs/ess/PAS2099/camera-trap-CVPR-configs"

yamls=()
batch_size=50

# Loop through each subdirectory in CONFIG_DIR
for subdir in "$CONFIG_DIR"/*/; do
    # Check if zs_lr0.000025.yaml exists in the subdirectory
    if [ -f "$subdir/zs_lr0.000025.yaml" ]; then
        YAML="$subdir/zs_lr0.000025.yaml"
        yamls+=("$YAML")
        
        if [ ${#yamls[@]} -eq $batch_size ]; then
            echo "Submitting batch of ${#yamls[@]} configs"
            sbatch eval.sh "${yamls[@]}"
            yamls=()
        fi
    fi
done

# Submit remaining configs
if [ ${#yamls[@]} -gt 0 ]; then
    echo "Submitting final batch of ${#yamls[@]} configs"
    sbatch eval.sh "${yamls[@]}"
fi