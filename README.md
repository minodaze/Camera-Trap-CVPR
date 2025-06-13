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
**SSF**

srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ssf  

**VPT-Deep**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --vpt_mode deep --vpt_num 10 

**VPT-Shallow**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --vpt_mode shallow --vpt_num 10    

**AdaptFormer**
 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_mlp_module adapter --ft_mlp_mode parallel --ft_mlp_ln before --adapter_bottleneck 64 --adapter_init lora_kaiming --adapter_scaler 0.1  

**Convpass**

  srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_attn_module convpass --ft_attn_mode parallel --ft_attn_ln after --ft_mlp_module convpass --ft_mlp_mode parallel --ft_mlp_ln after --convpass_scaler 0.1

**Adapter - VPT Version (Zero-init)**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init zero --adapter_scaler 1

**Pfeiffer Adapter - LoRA init**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init lora_xavier --adapter_scaler 1
       
**Pfeiffer Adapter - Random init**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1    

**Houlsby Adapter - Random init**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_attn_module adapter --ft_attn_mode sequential_after --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1       

**LoRA**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --lora_bottleneck 8 

**FacT-TK**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --optimizer adamw --fact_type tk --drop_path_rate 0.1  --fact_dim 32 --fact_scaler 0.1   

**Fact-TT**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --optimizer adamw --fact_type tt --drop_path_rate 0.1   

**ReAdapterv**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ft_attn_module repadapter --ft_attn_mode sequential_before --ft_mlp_module repadapter --ft_mlp_mode sequential_before --repadapter_bottleneck 8 --repadapter_scaler 10 

**BitFit**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --bitfit

**VQT**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --vqt_num 10

**Run LayerNorm Tuning**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --ln        

**DiffFit**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --difffit   

**Selective Attention Tuning**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --attention_index 9 10 11 --attention_type qkv 

**Selective MLP Tuning**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --mlp_index 9 10 11 --mlp_type full 

**Selective Block Tuning**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --block_index 9 10 11   

**Full Tuning**

 srun --pty --account=PAS2099 --job-name=bash --time=5:00:00 --nodes=1 --gpus-per-node=4 --mem=100G python run_pipeline.py --c '<.yaml>' --full --eval_freq 2    