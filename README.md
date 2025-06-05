```
<!-- 1. Data Preparation -->
mkdir data && cd data && ln -s /research/nfs_chao_209/zp/dataset/camera_trap_cropped data_cropped

<!-- 2. Model Preparation -->
mkdir pretrained_weight && cd pretrained_weight && git lfs clone https://huggingface.co/imageomics/bioclip

<!-- 3. Environment Preparation -->
pip install -r requirements.txt
```
