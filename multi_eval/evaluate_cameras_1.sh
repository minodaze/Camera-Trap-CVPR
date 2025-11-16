#!/bin/bash

# Script to evaluate cameras one by one from camera_list.txt

CAMERA_LIST="/fs/ess/PAS2099/sooyoung/Camera-Trap-CVPR/multi_eval/final.txt"
BASE_DIR="/fs/ess/PAS2099/sooyoung/Camera-Trap-CVPR"
LOGS_DIR1="/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/best_accum"
LOGS_DIR2="/fs/ess/PAS2099/sooyoung/camera-trap-CVPR-logs/accum_80/best_accum"

cd "$BASE_DIR" || exit 1

while IFS= read -r camera; do
    if [ -z "$camera" ]; then
        continue
    fi

    args_path1="$LOGS_DIR1/$camera/args.yaml"
    args_path2="$LOGS_DIR2/$camera/bioclip2/lora_8_text_head/all/log/args.yaml"

    if [ -f "$args_path1" ]; then
        args_path="$args_path1"
        loc="old"
    elif [ -f "$args_path2" ]; then
        args_path="$args_path2"
        loc="new"
    else
        echo "args.yaml not found for camera $camera"
        continue
    fi

    echo "Running evaluation for camera $camera with args $args_path ($loc)"
    python run_pipeline.py --c "$args_path"
done < "$CAMERA_LIST"