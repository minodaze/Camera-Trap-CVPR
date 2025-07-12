import torch
from torch import nn
from .utils import init_weight

class ConvPass(nn.Module):
    def __init__(self, dim, params=None):
        super().__init__()

        self.bottleneck = params.convpass_bottleneck
        
        # For vision: 2D convolution, for text: 1D convolution
        self.adapter_conv_2d = nn.Conv2d(self.bottleneck, self.bottleneck, 3, 1, 1)
        self.adapter_conv_1d = nn.Conv1d(self.bottleneck, self.bottleneck, 3, 1, 1)
        
        if params.convpass_xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv_2d.weight)
            nn.init.xavier_uniform_(self.adapter_conv_1d.weight)
        else:
            nn.init.zeros_(self.adapter_conv_2d.weight)
            nn.init.zeros_(self.adapter_conv_1d.weight)
            self.adapter_conv_2d.weight.data[:, :, 1, 1] += torch.eye(self.bottleneck, dtype=torch.float)
            self.adapter_conv_1d.weight.data[:, :, 1] += torch.eye(self.bottleneck, dtype=torch.float)
        
        nn.init.zeros_(self.adapter_conv_2d.bias)
        nn.init.zeros_(self.adapter_conv_1d.bias)

        self.adapter_down = nn.Linear(dim, self.bottleneck)
        self.adapter_up = nn.Linear(self.bottleneck, dim)
        init_weight(self.adapter_down, self.adapter_up, params.convpass_init)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.scale = params.convpass_scaler
        
        # Vision parameters
        if params.pretrained_weights == 'bioclip':
            crop_size, patch_size = 224, 16
        elif params.pretrained_weights == 'bioclip2':
            crop_size, patch_size = 224, 14
        else:
            crop_size, patch_size = params.image_size, params.patch_size
        self.patch_num = crop_size // patch_size
        self.params = params

    def _is_vision_input(self, x):
        """Detect if input is from vision or text transformer"""
        # Vision: expect ~257 tokens (256 patches + 1 cls), Text: expect ~77 tokens
        seq_len = x.shape[1] if len(x.shape) == 3 and x.shape[0] <= x.shape[1] else x.shape[0]
        return seq_len > 100  # Threshold to distinguish vision vs text

    def _forward_vision(self, x):
        """ConvPass for vision transformer (spatial convolution)"""
        # Handle OpenCLIP's LND format by transposing to BND
        if len(x.shape) == 3 and x.shape[0] > x.shape[1]:
            x = x.transpose(0, 1)
            transposed = True
        else:
            transposed = False

        B, N, C = x.shape

        x_down = self.adapter_down(x)
        x_down = self.act(x_down)

        # Spatial convolution for patches
        x_patch = x_down[:, 1:].reshape(B, self.patch_num, self.patch_num, self.bottleneck).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv_2d(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, self.patch_num * self.patch_num, self.bottleneck)

        # Handle CLS token
        x_cls = x_down[:, :1].reshape(B, 1, 1, self.bottleneck).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv_2d(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.bottleneck)

        x_down = torch.cat([x_cls, x_patch], dim=1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        x_up = x_up * self.scale * self.params.merge_factor
        
        if transposed:
            x_up = x_up.transpose(0, 1)
            
        return x_up

    def _forward_text(self, x):
        """ConvPass for text transformer (sequential convolution)"""
        # Handle OpenCLIP's LND format
        if len(x.shape) == 3 and x.shape[0] > x.shape[1]:
            x = x.transpose(0, 1)
            transposed = True
        else:
            transposed = False

        B, N, C = x.shape

        x_down = self.adapter_down(x)
        x_down = self.act(x_down)

        # Sequential 1D convolution
        x_conv = x_down.transpose(1, 2)  # B, C, N
        x_conv = self.adapter_conv_1d(x_conv)
        x_conv = x_conv.transpose(1, 2)  # B, N, C

        x_conv = self.act(x_conv)
        x_conv = self.dropout(x_conv)
        x_up = self.adapter_up(x_conv)

        x_up = x_up * self.scale * self.params.merge_factor
        
        if transposed:
            x_up = x_up.transpose(0, 1)
            
        return x_up

    def forward(self, x):
        if self._is_vision_input(x):
            return self._forward_vision(x)
        else:
            return self._forward_text(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
