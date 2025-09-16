#!/bin/bash

# Interactive Multi-GPU Training Script for ICICLE Benchmark
# This script allows you to configure each dataset with specific settings and GPU assignments

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

echo -e "${CYAN}🎯 ICICLE Benchmark Interactive GPU Training Configurator${NC}"
echo "=========================================================="

# Check if Python script exists
if [ ! -f "multi_gpu_training.py" ]; then
    echo -e "${RED}Error: multi_gpu_training.py not found in current directory${NC}"
    exit 1
fi

# ===================================================================
# HARDCODED CONFIGURATION
# ===================================================================
DATASET_INPUT_1="nz/nz_EFH_HCAMG12"
DATASET_INPUT_2="nz/nz_EFH_HCAMG12"
DATASET_INPUT_3="nz/nz_EFH_HCAMG12"
# DATASET_INPUT_4="MAD/MAD01"
CONDA_ENV="ICICLE"
WORKSPACE_ROOT="/fs/scratch/PAS2099/mino/ICICLE"
CONFIG_ROOT="/fs/scratch/PAS2099/camera-trap-final/configs"
# ===================================================================

# Build available datasets array dynamically from uncommented variables
AVAILABLE_DATASETS=()
DATASET_NAMES=()

if [ -n "$DATASET_INPUT_1" ]; then
    AVAILABLE_DATASETS+=("$DATASET_INPUT_1")
    DATASET_NAMES+=("Dataset 1: $DATASET_INPUT_1")
fi

if [ -n "$DATASET_INPUT_2" ]; then
    AVAILABLE_DATASETS+=("$DATASET_INPUT_2")
    DATASET_NAMES+=("Dataset 2: $DATASET_INPUT_2")
fi

if [ -n "$DATASET_INPUT_3" ]; then
    AVAILABLE_DATASETS+=("$DATASET_INPUT_3")
    DATASET_NAMES+=("Dataset 3: $DATASET_INPUT_3")
fi

if [ -n "$DATASET_INPUT_4" ]; then
    AVAILABLE_DATASETS+=("$DATASET_INPUT_4")
    DATASET_NAMES+=("Dataset 4: $DATASET_INPUT_4")
fi

# Training settings
TRAINING_SETTINGS=("lora_ce" "lora_bsm" "full_ce" "full_bsm")
SETTING_DESCRIPTIONS=("LoRA + CE" "LoRA + BSM" "Full + CE" "Full + BSM")

# GPU tracking
declare -a GPU_ASSIGNMENTS
declare -a SELECTED_DATASETS
declare -a SELECTED_DATASET_INDICES
declare -a SELECTED_SETTINGS
declare -a USED_GPUS

echo -e "${GREEN}📋 Available Datasets:${NC}"
for i in "${!DATASET_NAMES[@]}"; do
    echo "  $((i+1)). ${DATASET_NAMES[$i]}"
done

echo ""
echo -e "${BLUE}🔧 Available Training Settings:${NC}"
echo "  1. LoRA + CE  (LoRA with Cross Entropy loss)"
echo "  2. LoRA + BSM (LoRA with BSM loss)"
echo "  3. Full + CE  (Full training with Cross Entropy loss)"
echo "  4. Full + BSM (Full training with BSM loss)"

echo ""
echo -e "${YELLOW}💡 Instructions:${NC}"
echo "  - You can select 1-${#AVAILABLE_DATASETS[@]} datasets to train"
echo "  - Each dataset will run on a specific GPU (0-3)"
echo "  - Each dataset gets one training setting"
echo "  - No GPU can be used twice"

echo ""

# Function to display current configuration
display_config() {
    echo -e "${CYAN}📊 Current Configuration:${NC}"
    echo "=========================="
    if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
        echo "  No datasets configured yet"
    else
        for i in "${!SELECTED_DATASETS[@]}"; do
            dataset_converted=$(echo "${SELECTED_DATASETS[$i]}" | sed 's/\//_/g')
            slot_num=$((SELECTED_DATASET_INDICES[$i] + 1))
            echo "  GPU ${GPU_ASSIGNMENTS[$i]}: $dataset_converted (Slot $slot_num) → ${SETTING_DESCRIPTIONS[${SELECTED_SETTINGS[$i]}]}"
        done
    fi
    echo ""
}

# Function to check if a dataset slot is already selected
is_dataset_slot_selected() {
    local dataset_idx=$1
    for selected_idx in "${SELECTED_DATASET_INDICES[@]}"; do
        if [ "$selected_idx" == "$dataset_idx" ]; then
            return 0  # dataset slot is selected
        fi
    done
    return 1  # dataset slot is not selected
}

# Function to validate GPU availability
is_gpu_available() {
    local gpu=$1
    for used_gpu in "${USED_GPUS[@]}"; do
        if [ "$used_gpu" == "$gpu" ]; then
            return 1
        fi
    done
    return 0
}

# Main configuration loop
while true; do
    display_config
    
    max_datasets=${#AVAILABLE_DATASETS[@]}
    if [ ${#SELECTED_DATASETS[@]} -ge $max_datasets ]; then
        echo -e "${YELLOW}Maximum datasets ($max_datasets) reached!${NC}"
        break
    fi
    
    echo -e "${GREEN}Select a dataset to configure (or 'q' to finish, 'r' to reset):${NC}"
    for i in "${!DATASET_NAMES[@]}"; do
        if is_dataset_slot_selected "$i"; then
            echo "  $((i+1)). ${DATASET_NAMES[$i]} ✗ (already configured)"
        else
            echo "  $((i+1)). ${DATASET_NAMES[$i]} ✓ (available)"
        fi
    done
    echo "  q. Finish configuration"
    echo "  r. Reset configuration"
    
    max_choice=${#AVAILABLE_DATASETS[@]}
    read -p "Enter your choice (1-$max_choice, q, r): " dataset_choice
    
    case $dataset_choice in
        [1-9])
            if [ $dataset_choice -le ${#AVAILABLE_DATASETS[@]} ]; then
                dataset_idx=$((dataset_choice-1))
                selected_dataset="${AVAILABLE_DATASETS[$dataset_idx]}"
                
                # Check if dataset slot is already selected
                if is_dataset_slot_selected "$dataset_idx"; then
                    echo -e "${RED}Dataset slot already configured. Please choose a different dataset slot.${NC}"
                    echo ""
                    continue
                fi
                
                echo ""
                echo -e "${BLUE}Selected: ${DATASET_NAMES[$dataset_idx]}${NC}"
                echo ""
                echo "Choose training setting:"
                for i in "${!SETTING_DESCRIPTIONS[@]}"; do
                    echo "  $((i+1)). ${SETTING_DESCRIPTIONS[$i]}"
                done
                
                while true; do
                    read -p "Enter setting choice (1-4): " setting_choice
                    if [[ $setting_choice =~ ^[1-4]$ ]]; then
                        setting_idx=$((setting_choice-1))
                        break
                    else
                        echo -e "${RED}Invalid choice. Please enter 1-4.${NC}"
                    fi
                done
                
                echo ""
                echo -e "${YELLOW}Available GPUs:${NC}"
                for gpu in {0..3}; do
                    if is_gpu_available $gpu; then
                        echo "  $gpu ✓"
                    else
                        echo "  $gpu ✗ (already assigned)"
                    fi
                done
                
                while true; do
                    read -p "Select GPU (0-3): " gpu_choice
                    if [[ $gpu_choice =~ ^[0-3]$ ]]; then
                        if is_gpu_available $gpu_choice; then
                            break
                        else
                            echo -e "${RED}GPU $gpu_choice is already assigned. Please choose another.${NC}"
                        fi
                    else
                        echo -e "${RED}Invalid choice. Please enter 0-3.${NC}"
                    fi
                done
                
                # Add to configuration
                SELECTED_DATASETS+=("$selected_dataset")
                SELECTED_DATASET_INDICES+=("$dataset_idx")
                SELECTED_SETTINGS+=("$setting_idx")
                GPU_ASSIGNMENTS+=("$gpu_choice")
                USED_GPUS+=("$gpu_choice")
                
                echo -e "${GREEN}✓ Added: GPU $gpu_choice → $(echo "$selected_dataset" | sed 's/\//_/g') (Slot $((dataset_idx+1))) → ${SETTING_DESCRIPTIONS[$setting_idx]}${NC}"
                echo ""
            else
                echo -e "${RED}Invalid choice. Please enter 1-$max_choice, q, or r.${NC}"
            fi
            ;;
        [qQ])
            break
            ;;
        [rR])
            SELECTED_DATASETS=()
            SELECTED_DATASET_INDICES=()
            SELECTED_SETTINGS=()
            GPU_ASSIGNMENTS=()
            USED_GPUS=()
            echo -e "${YELLOW}Configuration reset!${NC}"
            echo ""
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
done

# Validate configuration
if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    echo -e "${RED}No datasets configured. Exiting.${NC}"
    exit 1
fi

echo ""
display_config

# Validate dataset directories exist
echo -e "${YELLOW}Validating dataset directories...${NC}"
for dataset in "${SELECTED_DATASETS[@]}"; do
    dataset_converted=$(echo "$dataset" | sed 's/\//_/g')
    if [ ! -d "$CONFIG_ROOT/$dataset_converted" ]; then
        echo -e "${RED}Error: Dataset directory not found: $CONFIG_ROOT/$dataset_converted${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} $dataset_converted"
done

echo ""
echo -e "${BLUE}📋 Training Commands that will be executed:${NC}"
echo "============================================="
for i in "${!SELECTED_DATASETS[@]}"; do
    dataset_converted=$(echo "${SELECTED_DATASETS[$i]}" | sed 's/\//_/g')
    gpu="${GPU_ASSIGNMENTS[$i]}"
    setting="${TRAINING_SETTINGS[${SELECTED_SETTINGS[$i]}]}"
    
    case $setting in
        "lora_ce")
            echo "  GPU $gpu: python run_pipeline.py --wandb --eval_per_epoch --test_per_epoch --save_best_model --c $CONFIG_ROOT/${dataset_converted}/${dataset_converted}_accu_ce.yaml --lora_bottleneck 8"
            ;;
        "lora_bsm")
            echo "  GPU $gpu: python run_pipeline.py --wandb --eval_per_epoch --test_per_epoch --save_best_model --c $CONFIG_ROOT/${dataset_converted}/${dataset_converted}_accu_bsm.yaml --lora_bottleneck 8"
            ;;
        "full_ce")
            echo "  GPU $gpu: python run_pipeline.py --wandb --eval_per_epoch --test_per_epoch --save_best_model --c $CONFIG_ROOT/${dataset_converted}/${dataset_converted}_accu_ce.yaml --full"
            ;;
        "full_bsm")
            echo "  GPU $gpu: python run_pipeline.py --wandb --eval_per_epoch --test_per_epoch --save_best_model --c $CONFIG_ROOT/${dataset_converted}/${dataset_converted}_accu_bsm.yaml --full"
            ;;
    esac
done

echo ""
read -p "Do you want to start training with this configuration? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo -e "${MAGENTA}🚀 Starting Interactive Multi-GPU Training...${NC}"
echo "Log file: interactive_gpu_training.log"
echo ""

# Create the training configuration for the Python script
TRAINING_CONFIGS=""
for i in "${!SELECTED_DATASETS[@]}"; do
    # Pass the original dataset name (with /) to Python script for conversion
    original_dataset="${SELECTED_DATASETS[$i]}"
    gpu="${GPU_ASSIGNMENTS[$i]}"
    setting="${TRAINING_SETTINGS[${SELECTED_SETTINGS[$i]}]}"
    
    if [ -n "$TRAINING_CONFIGS" ]; then
        TRAINING_CONFIGS="$TRAINING_CONFIGS,"
    fi
    TRAINING_CONFIGS="$TRAINING_CONFIGS$gpu:$original_dataset:$setting"
done

# Create a simple Python script to handle the interactive training
cat > interactive_training_runner.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_training(gpu_id, dataset, setting, config_root, workspace_root, conda_env):
    """Run training on a specific GPU with the given configuration"""
    
    # Set CUDA_VISIBLE_DEVICES for this process
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Convert dataset name from format like "MAD/MAD_MAD01" to "MAD_MAD_MAD01"
    dataset_converted = dataset.replace('/', '_')
    
    # Use the same name as directory for YAML file
    if setting in ['lora_ce', 'full_ce']:
        config_file = f"{config_root}/{dataset_converted}/{dataset_converted}_accu_ce.yaml"
    else:  # lora_bsm, full_bsm
        config_file = f"{config_root}/{dataset_converted}/{dataset_converted}_accu_bsm.yaml"
    
    # Build the command with default arguments
    cmd = [
        'conda', 'run', '-n', conda_env,
        'python', 'run_pipeline.py',
        '--wandb',
        '--eval_per_epoch', 
        '--test_per_epoch',
        '--save_best_model',
        '--c', config_file
    ]
    
    if setting.startswith('lora'):
        cmd.extend(['--lora_bottleneck', '8'])
    else:  # full training
        cmd.append('--full')
    
    print(f"🚀 Starting GPU {gpu_id}: {dataset_converted} with {setting}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {workspace_root}")
    print("-" * 50)
    
    # Run the training
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=None
        )
        
        if result.returncode == 0:
            print(f"✅ GPU {gpu_id} ({dataset_converted}, {setting}) completed successfully")
            return True, gpu_id, dataset_converted, setting, result.stdout, result.stderr
        else:
            print(f"❌ GPU {gpu_id} ({dataset_converted}, {setting}) failed with return code {result.returncode}")
            return False, gpu_id, dataset_converted, setting, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ GPU {gpu_id} ({dataset_converted}, {setting}) timed out")
        return False, gpu_id, dataset_converted, setting, "", "Training timed out"
    except Exception as e:
        print(f"💥 GPU {gpu_id} ({dataset_converted}, {setting}) failed with exception: {str(e)}")
        return False, gpu_id, dataset_converted, setting, "", str(e)

def main():
    if len(sys.argv) != 4:
        print("Usage: python interactive_training_runner.py <training_configs> <workspace_root> <conda_env>")
        sys.exit(1)
    
    training_configs = sys.argv[1]
    workspace_root = sys.argv[2]
    conda_env = sys.argv[3]
    config_root = "/fs/scratch/PAS2099/camera-trap-final/configs"
    
    # Parse training configurations
    training_tasks = []
    for config in training_configs.split(','):
        gpu_id, dataset, setting = config.split(':')
        training_tasks.append((int(gpu_id), dataset, setting))
    
    print(f"📋 Training Tasks:")
    for gpu_id, dataset, setting in training_tasks:
        dataset_converted = dataset.replace('/', '_')
        print(f"  GPU {gpu_id}: {dataset_converted} → {setting}")
    print("")
    
    # Run training tasks in parallel
    with ThreadPoolExecutor(max_workers=len(training_tasks)) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_training, gpu_id, dataset, setting, config_root, workspace_root, conda_env): (gpu_id, dataset, setting)
            for gpu_id, dataset, setting in training_tasks
        }
        
        # Wait for completion and collect results
        results = []
        for future in as_completed(future_to_task):
            gpu_id, dataset, setting = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"💥 Exception in GPU {gpu_id} ({dataset}, {setting}): {str(e)}")
                results.append((False, gpu_id, dataset, setting, "", str(e)))
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🏁 TRAINING SUMMARY")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for success, gpu_id, dataset, setting, stdout, stderr in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: GPU {gpu_id} - {dataset} ({setting})")
        if not success and stderr:
            print(f"  Error: {stderr[:200]}...")
        successful += 1 if success else 0
        failed += 1 if not success else 0
    
    print(f"\n📊 Results: {successful} successful, {failed} failed")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Run the interactive training
python interactive_training_runner.py "$TRAINING_CONFIGS" "$WORKSPACE_ROOT" "$CONDA_ENV" 2>&1 | tee interactive_gpu_training.log

exit_code=${PIPESTATUS[0]}

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}🎉 All training jobs completed successfully!${NC}"
else
    echo -e "${RED}⚠️  Some training jobs failed. Check the log for details.${NC}"
fi

echo "Full log saved to: interactive_gpu_training.log"

# Cleanup temporary script
rm -f interactive_training_runner.py
