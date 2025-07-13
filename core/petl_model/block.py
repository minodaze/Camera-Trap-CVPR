import torch.nn as nn
from timm.layers import DropPath
from timm.models.vision_transformer import LayerScale
from timm.layers.trace_utils import _assert
from .adapter import Adapter
from .convpass import ConvPass
from .repadapter import RepAdapter
from .ssf import init_ssf_scale_shift, ssf_ada
import torch
from typing import Optional
from .mlp import MlpPETL
from .attention import AttentionPETL

MODULE_REGISTRY = {
    'adapter': Adapter,
    'convpass': ConvPass,
    'repadapter': RepAdapter

}

class BlockPETL(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = MlpPETL,
            params=None,
            fact=None
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionPETL(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            ############# Added module #############
            params=params,
            fact=fact
            ############# Added module end #############
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            ############# Added module #############
            params=params,
            fact=fact
            ############# Added module end #############
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ############# Added module #############
        self.params = params
        if params.ft_attn_module:
            self.ft_attn_module = MODULE_REGISTRY[params.ft_attn_module](dim=dim, params=params)
        if params.ft_mlp_module:
            self.ft_mlp_module = MODULE_REGISTRY[params.ft_mlp_module](dim=dim, params=params)

        if self.params.ssf:
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

        if self.params.difffit:
            self.difffit_gamma1 = nn.Parameter(torch.ones(dim))
            self.difffit_gamma2 = nn.Parameter(torch.ones(dim))

        self.fact = fact
        ############# Added module end #############

    # Original forward method (easier to read and understand BUT NOT OPTIMIZED!!!!! MORE GPU MEMORY USAGE)
    def forward(self, x: torch.Tensor, idx) -> torch.Tensor:
        # MHSA path
        residual_attn = x

        if self.params.ssf:
            x_norm1 = ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1)
        else:
            x_norm1 = self.norm1(x)
        # ft attention module
        if self.params.ft_attn_module:
            if self.params.ft_attn_mode == 'parallel':
                x_original = self.drop_path1(self.ls1(self.attn(x_norm1, idx)))
                if self.params.ft_attn_ln == 'before':
                    x_ft_attn = self.drop_path1(self.ls1(self.ft_attn_module(x))) + x_original
                elif self.params.ft_attn_ln == 'after':
                    x_ft_attn = self.drop_path1(self.ls1(self.ft_attn_module(x_norm1))) + x_original
                else:
                    raise NotImplementedError
                del x_original
            elif self.params.ft_attn_mode == 'sequential_after':
                x_original = self.drop_path1(self.ls1(self.attn(x_norm1, idx)))
                x_ft_attn = self.drop_path1(self.ls1(self.ft_attn_module(x_original, add_residual=True)))
                del x_original
            elif self.params.ft_attn_mode == 'sequential_before':
                x_ft_attn = self.drop_path1(self.ls1(self.attn(self.ft_attn_module(x_norm1), idx)))
            else:
                raise NotImplementedError

            torch.cuda.empty_cache()
        else:
            # no tuning
            x_ft_attn = self.drop_path1(self.ls1(self.attn(x_norm1, idx)))

        # residual for attention module
        if self.params.difffit:
            x = self.difffit_gamma1 * x_ft_attn + residual_attn
        else:
            x = x_ft_attn + residual_attn

        del x_norm1, x_ft_attn, residual_attn
        torch.cuda.empty_cache()

        # MLP path
        residual_mlp = x

        if self.params.ssf:
            x_norm2 = ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2)
        else:
            x_norm2 = self.norm2(x)

        # ft mlp module
        if self.params.ft_mlp_module:
            if self.params.ft_mlp_mode == 'parallel':
                x_original = self.drop_path2(self.ls2(self.mlp(x_norm2, idx)))
                if self.params.ft_mlp_ln == 'before':
                    x_ft_mlp = self.drop_path2(self.ls2(self.ft_mlp_module(x))) + x_original
                elif self.params.ft_mlp_ln == 'after':
                    x_ft_mlp = self.drop_path2(self.ls2(self.ft_mlp_module(x_norm2))) + x_original
                else:
                    raise NotImplementedError
                del x_original
            elif self.params.ft_mlp_mode == 'sequential_after':
                x_original = self.drop_path2(self.ls2(self.mlp(x_norm2, idx)))
                x_ft_mlp = self.drop_path2(self.ls2(self.ft_mlp_module(x_original, add_residual=True)))
                del x_original
            elif self.params.ft_attn_mode == 'sequential_before':
                x_ft_mlp = self.drop_path2(self.ls2(self.mlp(self.ft_mlp_module(x_norm2), idx)))
            else:
                raise NotImplementedError

            torch.cuda.empty_cache()
        else:
            # no tuning
            x_ft_mlp = self.drop_path2(self.ls2(self.mlp(x_norm2, idx)))

        # residual for mlp module
        if self.params.difffit:
            x = self.difffit_gamma2 * x_ft_mlp + residual_mlp
        else:
            x = x_ft_mlp + residual_mlp
        del x_norm2, x_ft_mlp, residual_mlp
        torch.cuda.empty_cache()
        # Original forward
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def _get_norm(self, x, norm, order=0):
        if self.params.ssf and order == 1:
            return ssf_ada(norm(x), self.ssf_scale_1, self.ssf_shift_1)
        elif self.params.ssf and order == 2:
            return ssf_ada(norm(x), self.ssf_scale_2, self.ssf_shift_2)
        else:
            return norm(x)

    def _get_difffit(self, y, x, difffit_gamma):
        if self.params.difffit and difffit_gamma is not None:
            return difffit_gamma * y + x
        else:
            return y + x

    def _forward_helper(self, x, idx, norm, ft_module, dp, ls, main, order):
        if ft_module:
            if order == 1:
                ft_mode = self.params.ft_attn_mode
                ft_main = self.ft_attn_module
                if self.params.difffit:
                    difffit_gamma = self.difffit_gamma1
            elif order == 2:
                ft_mode = self.params.ft_mlp_mode
                ft_main = self.ft_mlp_module
                if self.params.difffit:
                    difffit_gamma = self.difffit_gamma2
            
            if ft_mode == 'parallel':
                if self.params.ft_attn_ln == 'before':
                    return self._get_difffit(dp(ls(ft_main(x))) + dp(ls(main(self._get_norm(x, norm, order), idx))), x, difffit_gamma)
                elif self.params.ft_attn_ln == 'after':
                    return self._get_difffit(dp(ls(ft_main(self._get_norm(x, norm, order)))) + dp(ls(main(self._get_norm(x, norm, order), idx))), x, difffit_gamma)
                else:
                    raise NotImplementedError
            elif ft_mode == 'sequential_after':
                return self._get_difffit(dp(ls(ft_main(dp(ls(main(self._get_norm(x, norm, order), idx))), add_residual=True))), x, difffit_gamma)
            elif ft_mode == 'sequential_before':
                return self._get_difffit(dp(ls(main(ft_main(self._get_norm(x, norm, order)), idx))), x, difffit_gamma)
            else:
                raise NotImplementedError
        else:
            if self.params.difffit and order== 1:
                difffit_gamma = self.difffit_gamma1
            elif self.params.difffit and order == 2:
                difffit_gamma = self.difffit_gamma2
            else:
                difffit_gamma = None
            # no tuning
            return self._get_difffit(dp(ls(main(self._get_norm(x, norm, order), idx))), x, difffit_gamma)

    # New forward method (optimized for GPU memory usage)
    def forward_new(self, x: torch.Tensor, idx) -> torch.Tensor:
        x = self._forward_helper(x, idx, self.norm1, self.params.ft_attn_module, self.drop_path1, self.ls1, self.attn, 1)
        x = self._forward_helper(x, idx, self.norm2, self.params.ft_mlp_module, self.drop_path2, self.ls2, self.mlp, 2)
        return x