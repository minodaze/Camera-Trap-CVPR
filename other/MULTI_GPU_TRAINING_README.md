# Multi-GPU Training Automation for ICICLE Benchmark

This system automates training across 4 GPUs, with each GPU handling one dataset through 4 different training settings.

## 🎯 Overview

- **4 GPUs** running in parallel
- **4 datasets** (one per GPU)
- **4 training settings** per dataset (16 total trainings)
- **Sequential execution** within each GPU
- **Automatic GPU memory management** between trainings

## 📋 Training Settings (in order)

Each GPU will execute these 4 training settings sequentially:

1. **LoRA BSM (bottleneck=8)**: `--c <dataset>_accu_bsm.yaml --lora_bottleneck 8`
2. **LoRA CE (bottleneck=8)**: `--c <dataset>_accu_ce.yaml --lora_bottleneck 8`  
3. **Full CE**: `--c <dataset>_accu_ce.yaml --full`
4. **Full BSM**: `--c <dataset>_accu_bsm.yaml --full`

## 🚀 Quick Start

### 1. Choose 4 Datasets

First, check available datasets:
```bash
ls config/pipeline/
```

Example datasets:
- `APN_K024`
- `APN_K034_new`
- `ENO_B06`
- `serengeti_C02`

### 2. Test Configuration (Recommended)

Test your configuration without running actual training:
```bash
./test_multi_gpu_config.sh APN_K024 APN_K034_new ENO_B06 serengeti_C02
```

### 3. Run Training

Launch the multi-GPU training:
```bash
./launch_multi_gpu_training.sh APN_K024 APN_K034_new ENO_B06 serengeti_C02
```

## 📁 File Structure

```
ICICLE-Benchmark/
├── multi_gpu_training.py          # Main automation script
├── launch_multi_gpu_training.sh   # Easy launcher script
├── test_multi_gpu_config.sh       # Configuration validator
├── MULTI_GPU_TRAINING_README.md   # This file
└── logs/                           # Training logs
    ├── gpu_0_<dataset>/           # Logs for GPU 0
    ├── gpu_1_<dataset>/           # Logs for GPU 1
    ├── gpu_2_<dataset>/           # Logs for GPU 2
    └── gpu_3_<dataset>/           # Logs for GPU 3
```

## 🔧 Advanced Usage

### Manual Python Execution

You can also run the Python script directly:
```bash
python multi_gpu_training.py \
    --datasets APN_K024 APN_K034_new ENO_B06 serengeti_C02 \
    --config-root /fs/scratch/PAS2099/camera-trap-final/configs \
    --workspace /fs/ess/PAS2099/sooyoung/ICICLE-Benchmark
```

### Options

- `--datasets`: Four dataset names (required)
- `--config-root`: Path to configuration files (default: `/fs/scratch/PAS2099/camera-trap-final/configs`)
- `--workspace`: Workspace path (default: `/fs/ess/PAS2099/sooyoung/ICICLE-Benchmark`)
- `--dry-run`: Validate configuration without running training

## 📊 Monitoring

### Real-time Monitoring

Monitor training progress in real-time:
```bash
# Watch main log
tail -f multi_gpu_training.log

# Watch specific GPU logs
tail -f logs/gpu_0_<dataset>/training_*.out
```

### Check GPU Usage

```bash
# Monitor GPU memory and usage
watch -n 1 nvidia-smi

# Check specific GPU
nvidia-smi -i 0,1,2,3
```

## 🛠️ Features

### Automatic GPU Memory Management
- Clears GPU memory between trainings
- Prevents memory leaks and CUDA out-of-memory errors
- Safe 5-second wait between memory clear and next training

### Comprehensive Logging
- Main log: `multi_gpu_training.log`
- Per-GPU logs: `logs/gpu_<id>_<dataset>/`
- Separate stdout/stderr files for each training
- Timestamped log files

### Error Handling
- Continues with remaining trainings if one fails
- Graceful shutdown on Ctrl+C
- Detailed error reporting
- Training summary with success/failure counts

### Process Management
- Each GPU runs in separate thread
- Proper process cleanup on interruption
- PID tracking for running processes
- Timeout handling for GPU memory clearing

## 📈 Training Execution Flow

```
GPU 0 (Dataset A)    GPU 1 (Dataset B)    GPU 2 (Dataset C)    GPU 3 (Dataset D)
     │                      │                      │                      │
     ├─ LoRA BSM (8)        ├─ LoRA BSM (8)        ├─ LoRA BSM (8)        ├─ LoRA BSM (8)
     ├─ Clear Memory        ├─ Clear Memory        ├─ Clear Memory        ├─ Clear Memory
     ├─ LoRA CE (8)         ├─ LoRA CE (8)         ├─ LoRA CE (8)         ├─ LoRA CE (8)
     ├─ Clear Memory        ├─ Clear Memory        ├─ Clear Memory        ├─ Clear Memory
     ├─ Full CE             ├─ Full CE             ├─ Full CE             ├─ Full CE
     ├─ Clear Memory        ├─ Clear Memory        ├─ Clear Memory        ├─ Clear Memory
     └─ Full BSM            └─ Full BSM            └─ Full BSM            └─ Full BSM
```

## ⚠️ Important Notes

### Configuration Requirements
- Config files must exist at: `/fs/scratch/PAS2099/camera-trap-final/configs/<dataset>/`
- Required files per dataset:
  - `<dataset>_accu_bsm.yaml`
  - `<dataset>_accu_ce.yaml`

### GPU Requirements
- 4 GPUs must be available (0, 1, 2, 3)
- Sufficient VRAM for your model and batch size
- CUDA-capable GPUs

### Workspace Requirements
- `run_pipeline.py` must be in the workspace root
- Write permissions for log directory creation
- Sufficient disk space for model checkpoints and logs

## 🔍 Troubleshooting

### Common Issues

**Config file not found:**
```bash
# Check if config files exist
ls /fs/scratch/PAS2099/camera-trap-final/configs/<dataset>/
```

**GPU memory issues:**
```bash
# Check GPU memory usage
nvidia-smi

# If needed, manually clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

**Permission issues:**
```bash
# Make scripts executable
chmod +x launch_multi_gpu_training.sh
chmod +x test_multi_gpu_config.sh
```

### Stopping Training

- **Graceful stop**: Press `Ctrl+C` once and wait
- **Force stop**: Press `Ctrl+C` multiple times
- **Kill processes**: `pkill -f run_pipeline.py`

## 📝 Example Output

```
[GPU-0][APN_K024] Starting training 1/4: LoRA BSM (bottleneck=8)
[GPU-1][APN_K034_new] Starting training 1/4: LoRA BSM (bottleneck=8)
[GPU-2][ENO_B06] Starting training 1/4: LoRA BSM (bottleneck=8)
[GPU-3][serengeti_C02] Starting training 1/4: LoRA BSM (bottleneck=8)
...
[GPU-0][APN_K024] Training 1 completed successfully!
[GPU-0][APN_K024] Clearing GPU memory...
[GPU-0][APN_K024] Starting training 2/4: LoRA CE (bottleneck=8)
...
```

## 🎉 Success Indicators

- All 16 trainings complete without errors
- Log files created for each training
- Model checkpoints saved in respective directories
- Final summary shows 16 completed, 0 failed trainings

## 📞 Support

If you encounter issues:
1. Check the main log: `multi_gpu_training.log`
2. Check GPU-specific logs: `logs/gpu_*/`
3. Verify configuration with dry-run mode
4. Ensure all dependencies are installed
5. Check GPU availability and memory
