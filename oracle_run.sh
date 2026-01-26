#!/bin/bash

# Read dataset paths from train_list.txt (one dataset per line)
# Check if train_list.txt exists
if [ ! -f "train_list.txt" ]; then
    echo "Error: train_list.txt not found!"
    echo "Please create train_list.txt with one dataset path per line."
    exit 1
fi

# Read datasets from file into array
readarray -t ALL_DATASETS < uselist/adapter.txt
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
        for ((i=start_index; i<=end_index; i++)); do
            job_datasets+=("${ALL_DATASETS[$i]}")
        done
        
        # Create a space-separated string of datasets for this job
        IFS=' ' datasets_string="${job_datasets[*]}"
        datasets_string="${datasets_string/_//}"
        
        job_counter=$((job_counter + 1))
        echo "Job $job_counter/$TOTAL_SUBMISSIONS: LR=$lr, Processing datasets ${start_index}-${end_index}"
        echo "  Datasets: ${datasets_string}"
        # sbatch script2/sbatch_run_adapter16_scaler1.0_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_sigmoid_beta0.9999_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_adapter_scaler0.1_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_adapter_scaler0.01_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_adapter_scaler2.0_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.1_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.15_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.2_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.3_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.4_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_vpt50_oracle.sh "${datasets_string}" "$lr"
        sbatch script2/sbatch_run_adapter32_scaler1.0_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_gamma0.0_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_gamma1.0_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.9999gamma2.0_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.9999gamma1.0_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.9999gamma1.8_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.9999gamma1.5_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.9999gamma1.2_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_sigmoid_beta0.999_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.999gamma1.8_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.999gamma1.5_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.999gamma1.2_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cb_focal_beta0.99_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.05_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.5_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_cdt0.7_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_bsm_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_lora_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_muti_gpu_bsm_oracle.sh "${datasets_string}" "$lr"
        # sbatch script2/sbatch_run_muti_gpu_lora_oracle.sh "${datasets_string}" "$lr"
        # sbatch script/sbatch_run_ub_lora_bsm_inter.sh "${datasets_string}" "$lr"
    done
done

echo "All $TOTAL_SUBMISSIONS jobs submitted!"
