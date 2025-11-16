#!/bin/bash

# Read datasets from file into array
readarray -t ALL_DATASETS < accum_resume_dataset_list.txt
readarray -t RESUME_MODEL < accum_resume_model_list.txt

# Remove empty lines and trim whitespace
TEMP_DATASETS=()
for dataset in "${ALL_DATASETS[@]}"; do
    # Trim whitespace and skip empty lines
    dataset=$(echo "$dataset" | xargs)
    if [[ -n "$dataset" && ! "$dataset" =~ ^[[:space:]]*# ]]; then
        TEMP_DATASETS+=("$dataset")
    fi
done
ALL_DATASETS=("${TEMP_DATASETS[@]}")

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
        job_models=()
        for ((i=start_index; i<=end_index; i++)); do
            job_datasets+=("${ALL_DATASETS[$i]}")
            job_models+=("${RESUME_MODEL[$i]}")
        done
        
        # Build dataset string (space-separated) after per-element underscore->slash conversion
        fixed_datasets=()
        for d in "${job_datasets[@]}"; do
            fixed_datasets+=("${d/_//}")
        done
        datasets_string="${fixed_datasets[*]}"

        # Build model-dirs string aligned to datasets; use '|' as delimiter to avoid conflicts with spaces
        model_dirs_string=""
        for m in "${job_models[@]}"; do
            if [[ -z "$model_dirs_string" ]]; then
                model_dirs_string="$m"
            else
                model_dirs_string+="|$m"
            fi
        done
        
        job_counter=$((job_counter + 1))
        echo "Job $job_counter/$TOTAL_SUBMISSIONS: LR=$lr, Processing datasets ${start_index}-${end_index}"
        echo "  Datasets: ${datasets_string}"
    echo "  Model Dirs: ${model_dirs_string}"

    sbatch script2/sbatch_run_resume_accum.sh "${datasets_string}" "$lr" "$model_dirs_string"
    done
done

echo "All $TOTAL_SUBMISSIONS jobs submitted!"
