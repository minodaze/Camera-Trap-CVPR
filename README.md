```
<!-- 1. Model Preparation -->
ln -s /fs/scratch/PAS2099/pretrained/pretrained_weights
To add more model for training, go to timm and download the pretrained weights and create a constructor method in core/petl_model/vision_transformer.py

<!-- 2. Data FOLDER IN OSC -->
/fs/scratch/PAS2099/camera-trap-cropped/datasets/
Do the soft link to the data folder, for example:
cd config/data && ln -s /fs/scratch/PAS2099/camera-trap-cropped/datasets/MAD/

<!-- 3. Environment Preparation -->
pip install -r requirements.txt
```

# RUN
**Text Encoder**
To activate text encoder for training, add --text full

**Text Encoder with LoRA**
srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-node=12 --mem=100G python run_pipeline.py --c '<.yaml>' --text lora --lora_bottleneck 8

**Full Fine Tune Text Encoder**
srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-node=12 --mem=100G python run_pipeline.py --c '<.yaml>' --text full --full

**Convpass**

  srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-node=12 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_attn_module convpass --ft_attn_mode parallel --ft_attn_ln after --ft_mlp_module convpass --ft_mlp_mode parallel --ft_mlp_ln after --convpass_scaler 0.1

**Adapter - VPT Version (Zero-init)**

 srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=12 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init zero --adapter_scaler 1

**Pfeiffer Adapter - LoRA init**

 srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=12 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init lora_xavier --adapter_scaler 1
       
**Pfeiffer Adapter - Random init**

 srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=12 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1    

**Houlsby Adapter - Random init**

 srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=12 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_attn_module adapter --ft_attn_mode sequential_after --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1       

**LoRA**

 srun --pty --account=PAS2099 --job-name=bash --time=4:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 --mem=64G python run_pipeline.py --c 'config/pipeline/ENO_C05/bsm/accumALFR.yaml' --eval_per_epoch --lora_bottleneck 8 


**ReAdapterv**

 srun --pty --account=PAS2099 --job-name=bash --time=24:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=12 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_attn_module repadapter --ft_attn_mode sequential_before --ft_mlp_module repadapter --ft_mlp_mode sequential_before --repadapter_bottleneck 8 --repadapter_scaler 10      

**Full Tuning**

 srun --pty --account=PAS2099 --job-name=bash --time=2:00:00 --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 --mem=100G python run_pipeline.py --c 'config/pipeline/ENO_C05/ce/accumulative-scratch.yaml' --plot_features --eval_per_epoch --full  