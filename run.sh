#!/bin/bash

# List of all dataset paths
# You can add up to 100 or more dataset paths here
ALL_DATASETS=(
  "swg/swg_loc_0182"
)

# Learning rates to test
# LEARNING_RATES=(0.000001 0.0000025 0.000005 0.00001 0.000025 0.00005 0.0001 0.00025 0.0005 )
LEARNING_RATES=(0.000025)
# Number of datasets to process per sbatch job
DATASETS_PER_JOB=4

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
        
        # Create a space-separated string of datasets for this job
        datasets_string="${job_datasets[*]}"
        
        job_counter=$((job_counter + 1))
        echo "Job $job_counter/$TOTAL_SUBMISSIONS: LR=$lr, Processing datasets ${start_index}-${end_index}"
        echo "  Datasets: ${datasets_string}"
        
        # Submit the sbatch job with the datasets and learning rate as arguments
        sbatch sbatch_run_1.sh "${datasets_string}" "$lr"
        
        # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
        sleep 1
    done
done

echo "All $TOTAL_SUBMISSIONS jobs submitted!"