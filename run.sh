#!/bin/bash

# List of all dataset paths
# You can add up to 100 or more dataset paths here
ALL_DATASETS=(
  "APN/APN_13U"
  "APN/APN_6U"
  "APN/APN_BOSP"
  "APN/APN_DW"
)



# Number of datasets to process per sbatch job
DATASETS_PER_JOB=6

# Calculate total number of jobs needed
TOTAL_DATASETS=${#ALL_DATASETS[@]}
TOTAL_JOBS=$(( (TOTAL_DATASETS + DATASETS_PER_JOB - 1) / DATASETS_PER_JOB ))

echo "Total datasets: $TOTAL_DATASETS"
echo "Datasets per job: $DATASETS_PER_JOB"
echo "Total jobs to submit: $TOTAL_JOBS"

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit jobs
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
    
    echo "Job $((job_index + 1))/$TOTAL_JOBS: Processing datasets ${start_index}-${end_index}"
    echo "  Datasets: ${datasets_string}"
    
    # Submit the sbatch job with the datasets as arguments
    sbatch sbatch_run_1.sh "${datasets_string}"
    
    # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done

echo "All $TOTAL_JOBS jobs submitted!"