#!/bin/bash

# List of all dataset paths
# You can add up to 100 or more dataset paths here
ALL_DATASETS=(
    "serengeti/serengeti_D04"
    "serengeti/serengeti_S13"
    "serengeti/serengeti_B07"
    "serengeti/serengeti_K04"
    "serengeti/serengeti_H03"
    "serengeti/serengeti_G01"
    "serengeti/serengeti_I12"
    "serengeti/serengeti_E10"
    "serengeti/serengeti_O04"
    "serengeti/serengeti_O03"
    "serengeti/serengeti_L01"
    "serengeti/serengeti_D11"
    "serengeti/serengeti_P04"
    "serengeti/serengeti_U09"
    "serengeti/serengeti_L06"
    "serengeti/serengeti_D06"
    "serengeti/serengeti_C10"
    "serengeti/serengeti_R05"
    "serengeti/serengeti_F11"
    "serengeti/serengeti_C01"
    "serengeti/serengeti_J05"
    "serengeti/serengeti_I10"
    "serengeti/serengeti_Q10"
    "serengeti/serengeti_L02"
    "serengeti/serengeti_G10"
    "serengeti/serengeti_U13"
    "serengeti/serengeti_N02"
    "serengeti/serengeti_E07"
    "serengeti/serengeti_O05"
    "serengeti/serengeti_L10"
    "serengeti/serengeti_H09"
    "serengeti/serengeti_K03"
    "serengeti/serengeti_K09"
    "serengeti/serengeti_E08"
    "serengeti/serengeti_C09"
    "serengeti/serengeti_E13"
    "serengeti/serengeti_E06"
    "serengeti/serengeti_B03"
    "serengeti/serengeti_N04"
    "serengeti/serengeti_C02"
    "serengeti/serengeti_T09"
    "serengeti/serengeti_E02"
    "serengeti/serengeti_F12"
    "serengeti/serengeti_P11"
    "serengeti/serengeti_G03"
    "serengeti/serengeti_K07"
    "serengeti/serengeti_C13"
    "serengeti/serengeti_L05"
    "serengeti/serengeti_N10"
    "serengeti/serengeti_C08"
    "serengeti/serengeti_J04"
    "serengeti/serengeti_L07"
    "serengeti/serengeti_F07"
    "serengeti/serengeti_K12"
    "serengeti/serengeti_R06"
    "serengeti/serengeti_P05"
    "serengeti/serengeti_M13"
    "serengeti/serengeti_G02"
    "serengeti/serengeti_H08"
    "serengeti/serengeti_R07"
    "serengeti/serengeti_F10"
    "serengeti/serengeti_L13"
    "serengeti/serengeti_L08"
    "serengeti/serengeti_N05"
    "serengeti/serengeti_N07"
    "serengeti/serengeti_M08"
    "serengeti/serengeti_T12"
    "serengeti/serengeti_F06"
    "serengeti/serengeti_O10"
    "serengeti/serengeti_G05"
    "serengeti/serengeti_O07"
    "serengeti/serengeti_N06"
    "serengeti/serengeti_D07"
    "serengeti/serengeti_S08"
    "serengeti/serengeti_H02"
    "serengeti/serengeti_I09"
    "serengeti/serengeti_P10"
    "serengeti/serengeti_J08"
    "serengeti/serengeti_Q04"
    "serengeti/serengeti_C12"
    "serengeti/serengeti_M07"
    "serengeti/serengeti_P12"
    "serengeti/serengeti_B08"
    "serengeti/serengeti_F03"
    "serengeti/serengeti_B09"
    "serengeti/serengeti_Q07"
    "serengeti/serengeti_I03"
    "serengeti/serengeti_H01"
    "serengeti/serengeti_D05"
    "serengeti/serengeti_J06"
    "serengeti/serengeti_C07"
    "serengeti/serengeti_D08"
    "serengeti/serengeti_H06"
    "serengeti/serengeti_E12"
    "serengeti/serengeti_R13"
    "serengeti/serengeti_I04"
    "serengeti/serengeti_E09"
    "serengeti/serengeti_S07"
    "serengeti/serengeti_G04"
    "serengeti/serengeti_B13"
    "serengeti/serengeti_G13"
    "serengeti/serengeti_F04"
    "serengeti/serengeti_I11"
    "serengeti/serengeti_C06"
    "serengeti/serengeti_S11"
    "serengeti/serengeti_J13"
    "serengeti/serengeti_L03"
    "serengeti/serengeti_Q08"
    "serengeti/serengeti_E05"
    "serengeti/serengeti_I06"
    "serengeti/serengeti_C04"
    "serengeti/serengeti_P08"
    "serengeti/serengeti_F02"
    "serengeti/serengeti_K08"
    "serengeti/serengeti_R08"
    "serengeti/serengeti_I05"
    "serengeti/serengeti_Q05"
    "serengeti/serengeti_K13"
    "serengeti/serengeti_E04"
    "serengeti/serengeti_I13"
    "serengeti/serengeti_Q12"
    "serengeti/serengeti_J02"
    "serengeti/serengeti_E03"
    "serengeti/serengeti_R12"
    "serengeti/serengeti_T10"
    "serengeti/serengeti_F05"
    "serengeti/serengeti_E01"
    "serengeti/serengeti_M05"
    "serengeti/serengeti_K02"
    "serengeti/serengeti_M03"
    "serengeti/serengeti_Q11"
    "serengeti/serengeti_V10"
    "serengeti/serengeti_M04"
    "serengeti/serengeti_D12"
    "serengeti/serengeti_H05"
    "serengeti/serengeti_G06"
    "serengeti/serengeti_D03"
    "serengeti/serengeti_T13"
    "serengeti/serengeti_Q09"
    "serengeti/serengeti_G12"
    "serengeti/serengeti_B12"
    "serengeti/serengeti_F13"
    "serengeti/serengeti_L09"
    "serengeti/serengeti_D13"
    "serengeti/serengeti_M10"
    "serengeti/serengeti_C11"
    "serengeti/serengeti_C03"
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
    sbatch sbatch_run.sh "${datasets_string}"
    
    # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done

echo "All $TOTAL_JOBS jobs submitted!"