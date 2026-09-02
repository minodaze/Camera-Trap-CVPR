import torch
checkpoint_path = 'pretrained_weights/wildclip/wildclip_vitb16_t1.pth'

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(checkpoint.keys())
print(checkpoint['state_dict'].keys())