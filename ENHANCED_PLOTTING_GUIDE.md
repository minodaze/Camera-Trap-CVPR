# Enhanced Per-Class Accuracy Extraction and Plotting Guide

This guide explains how to use the enhanced extraction and plotting tools for analyzing per-class accuracy from both new and completed ICICLE-Benchmark experiments.

## Tools Overview

1. **extract_per_class_from_completed.py** - Extract per-class metrics from completed experiments
2. **plot_checkpoint_accuracy_new.py** - Advanced plotting for specific checkpoints or aggregate results
3. **plot_per_class_accuracy.py** - General plotting tool for comparing experiments

## Core Features

### 1. Targeted Checkpoint Analysis for Accumulative Training

For accumulative training experiments, you can analyze specific checkpoints (like train_1, test_1, etc.):

#### Extract and Plot a Specific Checkpoint:
```bash
# Extract metrics for a specific checkpoint (e.g., ckp_1)
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_1" \
    --plot \
    --output_dir "extracted_metrics"

# Or extract first, then plot separately
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_1" \
    --output_dir "extracted_metrics"

python plot_checkpoint_accuracy_new.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_1" \
    --metrics_dir "extracted_metrics"
```

### 2. Aggregate Analysis for ZS/Upper_bound Experiments

For zero-shot (zs) and upper_bound experiments, analyze all checkpoints together:

#### Extract and Plot Aggregate Results:
```bash
# Extract all checkpoints and create aggregate plot
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name" \
    --plot \
    --plot_type "aggregate" \
    --output_dir "extracted_metrics"

# Or extract first, then plot aggregate
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name" \
    --output_dir "extracted_metrics"

python plot_checkpoint_accuracy_new.py \
    --experiment_path "log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name" \
    --metrics_dir "extracted_metrics" \
    --aggregate
```

### 3. Complete Analysis for Accumulative Training

Extract and plot all checkpoints individually for accumulative training:

```bash
# Extract and plot all checkpoints individually for accumulative training
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --plot \
    --output_dir "extracted_metrics"

# Or plot all checkpoints at once
python plot_checkpoint_accuracy_new.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --metrics_dir "extracted_metrics" \
    --all_checkpoints
```

## Detailed Usage Examples

### Example 1: Analyze Specific Checkpoint in Accumulative Training

You want to see the per-class accuracy for `ckp_1` checkpoint in an accumulative-scratch experiment:

```bash
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_1" \
    --plot \
    --output_dir "extracted_metrics"
```

This will:
- Extract metrics from `ckp_1_preds.pkl` (searching in experiment_path/, experiment_path/log/, or experiment_path/full/log/)
- Save metrics to `extracted_metrics/bioclip_2025-07-03-12-12-12_common_name_ckp_1_per_class_metrics.json`
- Generate a plot showing per-class accuracy for that specific checkpoint

### Example 2: Compare Different Checkpoints for Same Accumulative Experiment

```bash
# Extract checkpoint 1
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_1" \
    --output_dir "extracted_metrics"

# Extract checkpoint 5
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_5" \
    --output_dir "extracted_metrics"

# Plot both
python plot_checkpoint_accuracy_new.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_1" \
    --metrics_dir "extracted_metrics"

python plot_checkpoint_accuracy_new.py \
    --experiment_path "log/pipeline/ENO_E06/ce/accumulative-scratch/bioclip_2025-07-03-12-12-12_common_name" \
    --checkpoint "ckp_5" \
    --metrics_dir "extracted_metrics"
```

### Example 3: Aggregate Analysis for Zero-Shot Experiment

For experiments where you want to see overall performance across all checkpoints:

```bash
python extract_per_class_from_completed.py \
    --experiment_path "log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name" \
    --output_dir "extracted_metrics"

python plot_checkpoint_accuracy_new.py \
    --experiment_path "log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name" \
    --metrics_dir "extracted_metrics" \
    --aggregate
```

This creates:
- A bar plot showing mean per-class accuracy with error bars
- A heatmap showing per-class accuracy across all checkpoints
- Summary statistics in JSON format

## Output Files

### Extracted Metrics Files
- `{experiment_name}_{checkpoint}_per_class_metrics.json` - Individual checkpoint metrics
- `extraction_summary.json` - Overall summary of extraction

### Generated Plots
- `{experiment_name}_{checkpoint}_per_class_accuracy.png` - Individual checkpoint plot
- `{experiment_name}_aggregate_per_class_accuracy.png` - Aggregate plot with heatmap
- `{experiment_name}_aggregate_summary.json` - Aggregate statistics

## Advanced Options

### Custom Class Names
If class names are not found automatically:
```bash
python extract_per_class_from_completed.py \
    --experiment_path "path/to/experiment" \
    --class_names "Class1" "Class2" "Class3" \
    --checkpoint "train_1"
```

### Batch Processing (Legacy Mode)
For processing multiple experiments at once:
```bash
python extract_per_class_from_completed.py \
    --log_dir "log/pipeline" \
    --dataset "ENO_C05" \
    --method "focal/accumulative-scratch" \
    --output_dir "extracted_metrics"
```

## Integration with Existing Tools

The extracted metrics are compatible with the existing `plot_per_class_accuracy.py`:
```bash
python plot_per_class_accuracy.py \
    --log_dir "extracted_metrics" \
    --dataset "ENO_C05" \
    --methods "bioclip2_2025-07-06-22-13-54_scientific_name"
```

## Typical Workflows

### For Accumulative Training Analysis:
1. Extract specific checkpoints of interest
2. Plot individual checkpoints to see progression
3. Compare train vs test at same time points

### For ZS/Upper_bound Analysis:
1. Extract all checkpoints
2. Generate aggregate plots to see overall performance
3. Use heatmap to identify best-performing checkpoints

### For Comparative Analysis:
1. Extract metrics from multiple experiments
2. Use existing plotting tools to compare across methods
3. Generate publication-ready figures

This enhanced system provides flexible, targeted analysis capabilities while maintaining compatibility with existing workflows.
