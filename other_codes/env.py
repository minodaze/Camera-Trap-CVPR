import torch, platform, os
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())
# print("device:", torch.cuda.get_device_name(0))
print("python:", platform.python_version())
print("env AMP dtype:", os.getenv("TORCH_DTYPE", "default"))
