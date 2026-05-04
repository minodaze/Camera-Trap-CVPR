#!/bin/bash

# Read dataset paths from train_list.txt (one dataset per line)
# Check if train_list.txt exists
if [ ! -f "train_list.txt" ]; then
    echo "Error: train_list.txt not found!"
    echo "Please create train_list.txt with one dataset path per line."
    exit 1
fi

# Read datasets from file into array
readarray -t ALL_DATASETS < uselist/eval_dataset.txt

# Read model directories from model_dirs.txt (one directory per line)
readarray -t MODEL_DIRS < uselist/eval_model_path.txt

# Ratios to evaluate (Bash array syntax: no spaces around '=')
# ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# ratio=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ratio=(0.8)

# Remove empty lines and trim whitespace
TEMP_DATASETS=()
TEMP_DIRS=()
for dataset in "${ALL_DATASETS[@]}"; do
    # Trim whitespace and skip empty lines
    dataset=$(echo "$dataset" | xargs)
    if [[ -n "$dataset" && ! "$dataset" =~ ^[[:space:]]*# ]]; then
        TEMP_DATASETS+=("$dataset")
    fi
done
ALL_DATASETS=("${TEMP_DATASETS[@]}")

for model_dir in "${MODEL_DIRS[@]}"; do
    model_dir=$(echo "$model_dir" | xargs)
    if [[ -n "$model_dir" && ! "$model_dir" =~ ^[[:space:]]*# ]]; then
        TEMP_DIRS+=("$model_dir")
    fi
done
MODEL_DIRS=("${TEMP_DIRS[@]}")

if [ ${#ALL_DATASETS[@]} -ne ${#MODEL_DIRS[@]} ]; then
    echo "Error: The number of datasets (${#ALL_DATASETS[@]}) does not match the number of model directories (${#MODEL_DIRS[@]})."
    exit 1
fi

# Learning rates to test
# LEARNING_RATES=(0.000001 0.0000025 0.000005 0.00001 0.000025 0.00005 0.0001 0.00025 0.0005 )
LEARNING_RATES=(0.000025)
# Number of datasets to process per sbatch job
DATASETS_PER_JOB=1

# Calculate total number of jobs needed
TOTAL_DATASETS=${#ALL_DATASETS[@]}
TOTAL_LRS=${#LEARNING_RATES[@]}
TOTAL_JOBS=$(( (TOTAL_DATASETS + DATASETS_PER_JOB - 1) / DATASETS_PER_JOB ))
TOTAL_SUBMISSIONS=$((TOTAL_JOBS * TOTAL_LRS))

echo "Total datasets: $TOTAL_DATASETS"
echo "Total learning rates: $TOTAL_LRS"
echo "Datasets per job: $DATASETS_PER_JOB"
echo "Total jobs per LR: $TOTAL_JOBS"
echo "Total submissions: $TOTAL_SUBMISSIONS"

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit jobs
job_counter=0
for lr in "${LEARNING_RATES[@]}"; do
    echo "=== Submitting jobs for learning rate: $lr ==="
    
    for ((job_index=0; job_index<TOTAL_JOBS; job_index++)); do
        # Calculate start and end indices for this job
        start_index=$((job_index * DATASETS_PER_JOB))
        end_index=$(( start_index + DATASETS_PER_JOB - 1 ))
        
        # Make sure we don't exceed the array bounds
        if [ $end_index -ge $TOTAL_DATASETS ]; then
            end_index=$((TOTAL_DATASETS - 1))
        fi
        
        # Extract datasets for this job
        job_datasets=()
        for ((i=start_index; i<=end_index; i++)); do
            job_datasets+=("${ALL_DATASETS[$i]}")
        done
        job_model_dirs=()
        for ((i=start_index; i<=end_index; i++)); do
            if [ $i -lt ${#MODEL_DIRS[@]} ]; then
                job_model_dirs+=("${MODEL_DIRS[$i]}")
            else
                echo "Warning: Not enough model directories for dataset index $i. Using last available directory."
                job_model_dirs+=("${MODEL_DIRS[-1]}")
            fi
        done
        
        # Create a space-separated string of datasets for this job
        datasets_string="${job_datasets[*]}"
        
        job_counter=$((job_counter + 1))
        echo "Job $job_counter/$TOTAL_SUBMISSIONS: LR=$lr, Processing datasets ${start_index}-${end_index}"
        echo "  Datasets: ${datasets_string}"
        echo "  Model dirs: ${job_model_dirs[*]}"
        
        for i in "${!job_datasets[@]}"; do
            dataset="${job_datasets[$i]}"
            dataset="${dataset/_//}"
            model_dir="${job_model_dirs[$i]}"
            json_path="${model_dir}/final_training_summary.json"
            # Check if model directory exists
            if [ -f "$json_path" ]; then
                echo "  ✓ Model directory exists: $json_path"
                sbatch script2/sbatch_eval_oracle_ontrain.sh "${dataset}" "$lr" "${model_dir}"
            else
                echo "  ✗ Skipping ${dataset}: Model directory not found: $model_dir"
                continue
            fi
        done
        # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
    done
done

echo "All $TOTAL_SUBMISSIONS jobs submitted!"
