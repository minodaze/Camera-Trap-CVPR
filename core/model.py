import argparse
import logging
import math
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple

from contextlib import suppress
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
import json
import numpy as np
import os

import timm
from .petl_model.vision_transformer import VisionTransformerPETL
from .open_clip import create_model_and_transforms, get_cast_dtype, get_tokenizer

TUNE_MODULES = ['ft_attn_module', 'ft_mlp_module', 'head', 'vpt', 'ssf_scale', 'ssf_shift', 'lora', 'fact', 'vqt',
                'difffit']

OPENAI_IMAGENET_TEMPLATE = [
    'a photo of {CLZ_NAME}.',
    'a bad photo of a {CLZ_NAME}.',
    'a photo of many {CLZ_NAME}.',
    'a sculpture of a {CLZ_NAME}.',
    'a photo of the hard to see {CLZ_NAME}.',
    'a low resolution photo of the {CLZ_NAME}.',
    'a rendering of a {CLZ_NAME}.',
    'graffiti of a {CLZ_NAME}.',
    'a bad photo of the {CLZ_NAME}.',
    'a cropped photo of the {CLZ_NAME}.',
    'a tattoo of a {CLZ_NAME}.',
    'the embroidered {CLZ_NAME}.',
    'a photo of a hard to see {CLZ_NAME}.',
    'a bright photo of a {CLZ_NAME}.',
    'a photo of a clean {CLZ_NAME}.',
    'a photo of a dirty {CLZ_NAME}.',
    'a dark photo of the {CLZ_NAME}.',
    'a drawing of a {CLZ_NAME}.',
    'a photo of my {CLZ_NAME}.',
    'the plastic {CLZ_NAME}.',
    'a photo of the cool {CLZ_NAME}.',
    'a close-up photo of a {CLZ_NAME}.',
    'a black and white photo of the {CLZ_NAME}.',
    'a painting of the {CLZ_NAME}.',
    'a painting of a {CLZ_NAME}.',
    'a pixelated photo of the {CLZ_NAME}.',
    'a sculpture of the {CLZ_NAME}.',
    'a bright photo of the {CLZ_NAME}.',
    'a cropped photo of a {CLZ_NAME}.',
    'a plastic {CLZ_NAME}.',
    'a photo of the dirty {CLZ_NAME}.',
    'a jpeg corrupted photo of a {CLZ_NAME}.',
    'a blurry photo of the {CLZ_NAME}.',
    'a photo of the {CLZ_NAME}.',
    'a good photo of the {CLZ_NAME}.',
    'a rendering of the {CLZ_NAME}.',
    'a {CLZ_NAME} in a video game.',
    'a photo of one {CLZ_NAME}.',
    'a doodle of a {CLZ_NAME}.',
    'a close-up photo of the {CLZ_NAME}.',
    'a photo of a {CLZ_NAME}.',
    'the origami {CLZ_NAME}.',
    'the {CLZ_NAME} in a video game.',
    'a sketch of a {CLZ_NAME}.',
    'a doodle of the {CLZ_NAME}.',
    'a origami {CLZ_NAME}.',
    'a low resolution photo of a {CLZ_NAME}.',
    'the toy {CLZ_NAME}.',
    'a rendition of the {CLZ_NAME}.',
    'a photo of the clean {CLZ_NAME}.',
    'a photo of a large {CLZ_NAME}.',
    'a rendition of a {CLZ_NAME}.',
    'a photo of a nice {CLZ_NAME}.',
    'a photo of a weird {CLZ_NAME}.',
    'a blurry photo of a {CLZ_NAME}.',
    'a cartoon {CLZ_NAME}.',
    'art of a {CLZ_NAME}.',
    'a sketch of the {CLZ_NAME}.',
    'a embroidered {CLZ_NAME}.',
    'a pixelated photo of a {CLZ_NAME}.',
    'itap of the {CLZ_NAME}.',
    'a jpeg corrupted photo of the {CLZ_NAME}.',
    'a good photo of a {CLZ_NAME}.',
    'a plushie {CLZ_NAME}.',
    'a photo of the nice {CLZ_NAME}.',
    'a photo of the small {CLZ_NAME}.',
    'a photo of the weird {CLZ_NAME}.',
    'the cartoon {CLZ_NAME}.',
    'art of the {CLZ_NAME}.',
    'a drawing of the {CLZ_NAME}.',
    'a photo of the large {CLZ_NAME}.',
    'a black and white photo of a {CLZ_NAME}.',
    'the plushie {CLZ_NAME}.',
    'a dark photo of a {CLZ_NAME}.',
    'itap of a {CLZ_NAME}.',
    'graffiti of the {CLZ_NAME}.',
    'a toy {CLZ_NAME}.',
    'itap of my {CLZ_NAME}.',
    'a photo of a cool {CLZ_NAME}.',
    'a photo of a small {CLZ_NAME}.',
    'a tattoo of the {CLZ_NAME}.',
]

BIOCLIP_TEMPLATE = [
    'a photo of {CLZ_NAME}.',
]
CAMERA_TRAP_TEMPLATE = [
    'a camera trap photo of {CLZ_NAME}.',
]


class HFVisualEncoder(nn.Module):
    """Adapter to make HF SigLIP/SigLIP2 vision encoders look like an OpenCLIP visual model."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            out = self.model.get_image_features(pixel_values=pixel_values)
            return _hf_pooled_features(out)
        out = self.model(pixel_values=pixel_values)
        return _hf_pooled_features(out)


def _hf_pooled_features(output: object) -> torch.Tensor:
    """Extract a single pooled feature vector from a HF model output."""
    if torch.is_tensor(output):
        return output

    # Generic encoder outputs
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]  # CLS by convention

    raise RuntimeError(f"HF model output does not contain pooled features: {type(output)}")


def hf_get_text_features(model: nn.Module, inputs: dict) -> torch.Tensor:
    """Get projected text features for HF CLIP/SigLIP(-2) style models.

    Prefers `model.get_text_features()` when it returns a Tensor. If it returns a full
    output object instead (e.g. BaseModelOutputWithPooling), we extract pooled features
    and apply a projection layer when available.
    """

    if hasattr(model, "get_text_features"):
        out = model.get_text_features(**inputs)
        if torch.is_tensor(out):
            return out
        # Some implementations return a model output; fall through to extraction.
        try:
            pooled = _hf_pooled_features(out)
        except Exception:
            pooled = None
        if pooled is not None:
            if hasattr(model, "text_projection") and isinstance(model.text_projection, nn.Module):
                return model.text_projection(pooled)
            if hasattr(model, "text_proj") and isinstance(model.text_proj, nn.Module):
                return model.text_proj(pooled)
            return pooled

    # Manual path via underlying text encoder
    if hasattr(model, "text_model"):
        out = model.text_model(**inputs)
        pooled = _hf_pooled_features(out)
        if hasattr(model, "text_projection") and isinstance(model.text_projection, nn.Module):
            return model.text_projection(pooled)
        if hasattr(model, "text_proj") and isinstance(model.text_proj, nn.Module):
            return model.text_proj(pooled)
        return pooled

    # Last resort: try the main forward
    out = model(**inputs)
    return _hf_pooled_features(out)


def infer_hf_embed_dim(model: nn.Module, processor=None) -> int:
    """Infer the embedding dimension for HF SigLIP/SigLIP2-style models.

    Different checkpoints/configs expose this as `projection_dim`, `projection_size`, or only
    via nested `text_config`/`vision_config`. As a last resort, we run a tiny dummy forward
    pass with text inputs to infer the final feature dimension.
    """

    cfg = getattr(model, "config", None)
    checked = []

    def _read_attr(obj, name: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _as_int(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None

    # Common config fields across HF CLIP-like models
    field_names = (
        "projection_dim",
        "projection_size",
        "embed_dim",
        "hidden_size",
        "d_model",
    )
    for scope_name, scope_obj in (
        ("config", cfg),
        ("config.text_config", _read_attr(cfg, "text_config")),
        ("config.vision_config", _read_attr(cfg, "vision_config")),
    ):
        for field in field_names:
            value = _read_attr(scope_obj, field)
            checked.append(f"{scope_name}.{field}")
            dim = _as_int(value)
            if dim is not None and dim > 0:
                return dim

    # Fallback: infer from a tiny forward pass on text
    if processor is not None:
        try:
            device = next(model.parameters()).device
            with torch.no_grad():
                inputs = processor(text=["test"], padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if hasattr(model, "get_text_features"):
                    feats = model.get_text_features(**inputs)
                else:
                    out = model(**inputs)
                    if hasattr(out, "text_embeds") and out.text_embeds is not None:
                        feats = out.text_embeds
                    else:
                        feats = getattr(out, "pooler_output", None)
                if feats is not None and hasattr(feats, "shape") and feats.ndim >= 2:
                    return int(feats.shape[-1])
        except Exception:
            pass

    raise RuntimeError(
        "Could not infer HF embedding dimension from model config. "
        f"Checked: {', '.join(checked)}"
    )


def get_class_embedding_hf(model: nn.Module, processor, embed_dim: int, class_name_idx, text_template: str = "openai"):
    """Compute normalized per-class text embeddings using a HF model/processor.

    This mirrors `get_class_embedding()` for OpenCLIP, but uses `processor` + HF forward.
    """

    device = next(model.parameters()).device
    with torch.no_grad():
        class_embedding = torch.empty(len(class_name_idx), embed_dim)
        for class_name, class_idx in class_name_idx.items():
            texts = get_texts(class_name, text_template)
            inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            text_feats = hf_get_text_features(model, inputs)

            if not torch.is_tensor(text_feats):
                raise RuntimeError(f"HF text features must be a Tensor, got: {type(text_feats)}")
            if text_feats.ndim == 1:
                text_feats = text_feats.unsqueeze(0)
            if text_feats.shape[-1] != embed_dim:
                raise RuntimeError(
                    f"HF text feature dim ({int(text_feats.shape[-1])}) does not match embed_dim ({int(embed_dim)}). "
                    "This usually indicates the wrong config field was used (projection vs hidden size) or the model "
                    "returned unprojected pooled features."
                )

            text_feats = F.normalize(text_feats, dim=-1).mean(dim=0)
            text_feats = F.normalize(text_feats, dim=-1)
            class_embedding[class_idx] = text_feats.detach().cpu()
    return class_embedding

def _module_param_dtype(module: nn.Module) -> torch.dtype:
    for p in module.parameters(recurse=True):
        return p.dtype
    return torch.float32

class CLIPClassifier(nn.Module):
    def __init__(self, visual_model, hidden_size, device):
        super(CLIPClassifier, self).__init__()
        self.visual_model = visual_model
        self.head = None
        self.initialized = False
        self.device = device
        self.init_text = False

    def init_head(self, class_embedding):
        self.head = nn.Linear(class_embedding.size(1), class_embedding.size(0), bias=True)
        self.head.weight.data = class_embedding
        self.head.bias.data.zero_()
        self.initialized = True
    
    def set_text(self, text_model, tokenizer, text_embed_dim, class_name_idx, template):
        """Set the text model and tokenizer for the classifier."""
        self.text_model = text_model
        self.add_module('text_model', self.text_model)  # This registers the text model as a submodule
        self.tokenizer = tokenizer
        self.text_embed_dim = text_embed_dim
        self.class_name_idx = class_name_idx
        self.text_template = template
        self.init_text = True
    
    def set_proj_head(self, proj_head):
        """Set a custom projection head for the classifier."""
        assert isinstance(proj_head, nn.Module), "proj_head must be an instance of nn.Module"
        self.proj_head = proj_head
        for param in self.proj_head.parameters():
            param.requires_grad = True
    
    def proj_features(self, images):
        """Project the features using the custom projection head."""
        if not hasattr(self, 'proj_head'):
            raise RuntimeError("Projection head is not set.")
        feats = self.forward_features(images)
        return F.normalize(self.proj_head(feats), dim=1)
    
    def forward(self, images, return_feats=False):
        """Forward pass of the classifier."""
        # import pdb; pdb.set_trace()
        x = self.visual_model(images)
        feats = F.normalize(x, dim=-1)  # Normalize the features
        
        if self.init_text:
            class_embedding = self.get_class_embedding(self.text_model, self.tokenizer, self.text_embed_dim, self.class_name_idx, self.text_template).to(self.device)
            x = F.linear(feats, class_embedding, bias=None)  # Use the class embedding to compute logits
            del class_embedding  # Free memory
        elif self.initialized:
            x = self.head(feats)
        else:
            raise RuntimeError("Forward pass requires either text model or initialized head.")
        if return_feats:
            return x, feats
        else:
            return x
    
    def forward_features(self, images):
        """Forward pass to get the features from the visual model."""
        x = self.visual_model(images)
        feats = F.normalize(x, dim=-1)
        return feats

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            logging.info(f'Creating directory {os.path.dirname(path)}... ')
            os.makedirs(os.path.dirname(path))
        logging.info(f'Saving classifier to {path}... ')
        torch.save(self.state_dict(), path)

    def load(self, path):
        logging.info(f'Loading classifier from {path}... ')
        self.load_state_dict(torch.load(path))
    
    def interpolate_head(self, model, alpha=0.5):
        """Interpolate the class embedding with the current head weights."""
        if not self.initialized:
            raise RuntimeError("Head is not initialized. Cannot interpolate.")
        if model.head.weight.size(1) != self.head.weight.size(1):
            raise ValueError("Class embedding dimension must match head weight dimension.")
        if model.head.weight.size(0) != self.head.weight.size(0):
            raise ValueError("Class embedding size must match head weight size.")
        
        # Interpolate the weights
        new_weight = (1 - alpha) * self.head.weight.data + alpha * model.head.weight.data
        self.head.weight.data = new_weight
        self.initialized = True
    
    def interpolate_model(self, classifier, alpha):
        """
        Interpolate the current model's parameters with another CLIPClassifier.
            
        Args:
            classifier: Another CLIPClassifier instance to interpolate with
            alpha: Interpolation factor (0.0 = keep current model, 1.0 = use other model)
        """
        if not isinstance(classifier, CLIPClassifier):
            raise ValueError("classifier must be an instance of CLIPClassifier.")
            
        # Interpolate visual model parameters
        for (name1, param1), (name2, param2) in zip(self.visual_model.named_parameters(), classifier.visual_model.named_parameters()):
            if name1 != name2:
                raise ValueError(f"Parameter names don't match: {name1} vs {name2}")
            param1.data = (1 - alpha) * param1.data + alpha * param2.data
            
        # Interpolate head parameters if both are initialized
        if self.initialized and classifier.initialized:
            self.head.weight.data = (1 - alpha) * self.head.weight.data + alpha * classifier.head.weight.data
            self.head.bias.data = (1 - alpha) * self.head.bias.data + alpha * classifier.head.bias.data
            
        # Interpolate text model parameters if both have text models
        if hasattr(self, 'text_model') and hasattr(classifier, 'text_model'):
            for (name1, param1), (name2, param2) in zip(self.text_model.named_parameters(), classifier.text_model.named_parameters()):
                if name1 != name2:
                    raise ValueError(f"Text model parameter names don't match: {name1} vs {name2}")
                param1.data = (1 - alpha) * param1.data + alpha * param2.data
            
        # Interpolate projection head if both have it
        if hasattr(self, 'proj_head') and hasattr(classifier, 'proj_head'):
            for (name1, param1), (name2, param2) in zip(self.proj_head.named_parameters(), classifier.proj_head.named_parameters()):
                if name1 != name2:
                    raise ValueError(f"Projection head parameter names don't match: {name1} vs {name2}")
                param1.data = (1 - alpha) * param1.data + alpha * param2.data        

    def get_class_embedding(self, model, tokenizer, embed_dim, class_name_idx, text_template='openai'): 
        device = next(model.parameters()).device
        context_length = model.context_length
        class_embedding = torch.empty(len(class_name_idx), embed_dim)
        for class_name, class_idx in class_name_idx.items():
            # logging.info(f'Getting class embedding for {class_name}... ')
            texts = self.get_texts(class_name, text_template)
            # logging.info('Texts: ')
            # for t in texts:
            #     logging.info(f'\t{t}')
            texts = tokenizer(
                    texts, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=context_length, 
                    return_tensors='pt'
                )
            input_ids = texts['input_ids'].to(device)
            _class_embedding = model.encode_text(input_ids)
            _class_embedding = F.normalize(_class_embedding, dim=-1).mean(dim=0)
            _class_embedding = F.normalize(_class_embedding, dim=-1)
            class_embedding[class_idx] = _class_embedding
        return class_embedding

    def get_texts(self, c, text_template='openai'):
        texts = [template.format(CLZ_NAME=c) for template in BIOCLIP_TEMPLATE]
        return texts
    
    def interpolate_head(self, model, alpha=0.5):
        """Interpolate the class embedding with the current head weights."""
        if not self.initialized:
            raise RuntimeError("Head is not initialized. Cannot interpolate.")
        if model.head.weight.size(1) != self.head.weight.size(1):
            raise ValueError("Class embedding dimension must match head weight dimension.")
        if model.head.weight.size(0) != self.head.weight.size(0):
            raise ValueError("Class embedding size must match head weight size.")
        
        # Interpolate the weights
        new_weight = (1 - alpha) * self.head.weight.data + alpha * model.head.weight.data
        self.head.weight.data = new_weight
        self.initialized = True
    
    def interpolate_model(self, classifier, alpha):
        """
        Interpolate the current model's parameters with another CLIPClassifier.
            
        Args:
            classifier: Another CLIPClassifier instance to interpolate with
            alpha: Interpolation factor (0.0 = keep current model, 1.0 = use other model)
        """
        if not isinstance(classifier, CLIPClassifier):
            raise ValueError("classifier must be an instance of CLIPClassifier.")
            
        # Interpolate visual model parameters
        for (name1, param1), (name2, param2) in zip(self.visual_model.named_parameters(), classifier.visual_model.named_parameters()):
            if name1 != name2:
                raise ValueError(f"Parameter names don't match: {name1} vs {name2}")
            param1.data = alpha * param1.data + (1 - alpha) * param2.data
            
        # Interpolate head parameters if both are initialized
        if self.initialized and classifier.initialized:
            self.head.weight.data = alpha * self.head.weight.data + (1 - alpha) * classifier.head.weight.data
            self.head.bias.data = alpha * self.head.bias.data + (1 - alpha) * classifier.head.bias.data
            
        # Interpolate text model parameters if both have text models
        if hasattr(self, 'text_model') and hasattr(classifier, 'text_model'):
            for (name1, param1), (name2, param2) in zip(self.text_model.named_parameters(), classifier.text_model.named_parameters()):
                if name1 != name2:
                    raise ValueError(f"Text model parameter names don't match: {name1} vs {name2}")
                param1.data = alpha * param1.data + (1 - alpha) * param2.data

        # Interpolate projection head if both have it
        if hasattr(self, 'proj_head') and hasattr(classifier, 'proj_head'):
            for (name1, param1), (name2, param2) in zip(self.proj_head.named_parameters(), classifier.proj_head.named_parameters()):
                if name1 != name2:
                    raise ValueError(f"Projection head parameter names don't match: {name1} vs {name2}")
                param1.data = alpha * param1.data + (1 - alpha) * param2.data

    def interpolate_lora(self, classifier, alpha=0.5):
        """Interpolate the LoRA parameters in the visual model."""
        for block in self.visual_model.transformer.resblocks:
            block.attn.lora.merge_factor = alpha
        self.head.weight.data = alpha * self.head.weight.data + (1 - alpha) * classifier.head.weight.data
        self.head.bias.data = alpha * self.head.bias.data + (1 - alpha) * classifier.head.bias.data

    def get_class_embedding(self, model, tokenizer, embed_dim, class_name_idx, text_template='openai'): 
        device = next(model.parameters()).device
        context_length = model.context_length
        class_embedding = torch.empty(len(class_name_idx), embed_dim)
        for class_name, class_idx in class_name_idx.items():
            # logging.info(f'Getting class embedding for {class_name}... ')
            texts = self.get_texts(class_name, text_template)
            # logging.info('Texts: ')
            # for t in texts:
            #     logging.info(f'\t{t}')
            texts = tokenizer(
                    texts, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=context_length, 
                    return_tensors='pt'
                )
            input_ids = texts['input_ids'].to(device)
            _class_embedding = model.encode_text(input_ids)
            _class_embedding = F.normalize(_class_embedding, dim=-1).mean(dim=0)
            _class_embedding = F.normalize(_class_embedding, dim=-1)
            class_embedding[class_idx] = _class_embedding
        return class_embedding

def build_classifier(params, class_name_idx, device): 
    if isinstance(class_name_idx, list):
        class_name_idx = {c: i for i, c in enumerate(class_name_idx)}
    class_num = len(class_name_idx)
        
    # Log initial GPU memory before model loading
    if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
        from utils.gpu_monitor import log_gpu_memory
        log_gpu_memory("model_build", "before_bioclip_load", device=device, enable_wandb=getattr(params, 'wandb', False))
    
    # SigLIP2 (HF) path: zero-shot only, using precomputed text embeddings as linear head
    if params.pretrained_weights == 'siglip2':
        if getattr(params, 'text', 'head') != 'head':
            raise ValueError("SigLIP2 currently supports only --text head (zero-shot head embeddings)")

        from transformers import AutoModel, AutoProcessor

        siglip2_dir = getattr(params, 'siglip2_dir', None) or 'pretrained_weights/siglip2-large-patch16-256'
        if not isinstance(siglip2_dir, (str, os.PathLike)) or str(siglip2_dir).strip() == "":
            raise ValueError(
                "Invalid --siglip2_dir. Provide a local directory path, e.g. "
                "--siglip2_dir pretrained_weights/siglip2-large-patch16-256"
            )
        # import pdb; pdb.set_trace()
        logging.info(f"Using SigLIP2 model from local dir: {siglip2_dir}")

        processor = AutoProcessor.from_pretrained(siglip2_dir)
        # torch_dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
        siglip2 = AutoModel.from_pretrained(siglip2_dir)
        siglip2 = siglip2.to(device)
        siglip2.eval()

        try:
            embed_dim = infer_hf_embed_dim(siglip2, processor=processor)
        except Exception as e:
            raise RuntimeError(
                f"Could not infer SigLIP2 embedding dimension for --siglip2_dir={siglip2_dir}. "
                "This checkpoint likely stores the dim under nested config fields (e.g. text_config.projection_size). "
                f"Original error: {e}"
            )

        visual_encoder = HFVisualEncoder(siglip2)
        classifier = CLIPClassifier(visual_encoder, embed_dim, device)

        class_embedding = get_class_embedding_hf(
            siglip2,
            processor,
            embed_dim,
            class_name_idx,
            text_template=getattr(params, 'text_template', 'openai'),
        )
        print(f"Class embedding shape: {class_embedding.shape}")
        print(f"Class embedding sample (first 5 values): {class_embedding[0][:10]}")
        # import pdb; pdb.set_trace()
        classifier.init_head(class_embedding)

        # Freeze by default unless explicitly fine-tuning
        for name, parameter in siglip2.named_parameters():
            if getattr(params, 'full', False):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

        classifier = classifier.to(device)

        if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
            from utils.gpu_monitor import log_gpu_memory
            log_gpu_memory("model_build", "final", device=device, enable_wandb=getattr(params, 'wandb', False))

        return classifier

    # Load the BIOCLIP model to get the class embeddings
    if params.pretrained_weights == 'bioclip':
        logging.info("Using Bioclip model. ")
        bioclip_model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-B-16',
                'pretrained_weights/bioclip/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=None,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                aug_cfg={},
                output_dict=True,
            )
        tokenizer = AutoTokenizer.from_pretrained('pretrained_weights/bioclip')
    elif params.pretrained_weights == 'bioclip2':
        logging.info("Using Bioclip-2 model. ")
        weight_path = 'pretrained_weights/bioclip-2/open_clip_pytorch_model.bin'
        bioclip_model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-L-14',
            weight_path,
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True,
            params=params
        )
        tokenizer = AutoTokenizer.from_pretrained('pretrained_weights/bioclip-2')
    elif params.pretrained_weights == 'openai-ViT-L-14':
        logging.info("Using OpenAI ViT-L-14 model. ")
        bioclip_model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-L-14',
            'openai',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True,
            params=params
        )
        tokenizer = AutoTokenizer.from_pretrained('pretrained_weights/clip-vit-large-patch14')
    else:
        raise NotImplementedError(f"Pretrained weights {params.pretrained_weights} not supported. ")
    
    # Log memory after loading BIOCLIP
    if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
        log_gpu_memory("model_build", "after_bioclip_load", device=device, enable_wandb=getattr(params, 'wandb', False))
    
    # del bioclip_model.visual
    
    # Log memory after deleting visual model
    if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
        log_gpu_memory("model_build", "after_visual_delete", device=device, enable_wandb=getattr(params, 'wandb', False))
    
    # Get the model and tune parameters
    # model, tune_parameters, model_grad_params_no_head = get_model(params, class_num, bioclip_model)
    
    # Log memory after getting PETL model
    if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
        log_gpu_memory("model_build", "after_petl_model", device=device, enable_wandb=getattr(params, 'wandb', False))

    ###################################################################
    classifier = CLIPClassifier(bioclip_model.visual, bioclip_model.embed_dim, device)

    text_embed_dim = bioclip_model.embed_dim
    if params.text == 'head':
        class_embedding = get_class_embedding(bioclip_model, tokenizer, text_embed_dim, class_name_idx, text_template=params.text_template)
        classifier.init_head(class_embedding)

    else:
        classifier.set_text(bioclip_model, tokenizer, text_embed_dim, class_name_idx, params.text_template)
        
    for name, parameter in bioclip_model.named_parameters():
        if params.full:
            parameter.requires_grad = True
            if params.debug:
                logging.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
        else:
            if any(m in name for m in TUNE_MODULES):
                parameter.requires_grad = True
                if params.debug:
                    logging.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
            else:
                parameter.requires_grad = False

    for name, parameter in classifier.named_parameters():
        if parameter.requires_grad:
            logging.info("Classifier trainable param: {}, {}, {}".format(name, parameter.numel(), parameter.shape))

    # Log memory after class embedding
    if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
        log_gpu_memory("model_build", "after_class_embedding", device=device, enable_wandb=getattr(params, 'wandb', False))
    classifier = classifier.to(device)

    # if params.model_path != 'None':
    #     classifier.load(params.model_path)
    
    # Log final memory usage
    if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
        log_gpu_memory("model_build", "final", device=device, enable_wandb=getattr(params, 'wandb', False))
    
    return classifier

# _LOOKUP_PATH = 'config/common_name_lookup.json'
# _lookup = json.load(open(_LOOKUP_PATH))

def get_texts(c, text_template='openai'):
    if text_template == 'customized':
        texts = [template.format(CLZ_NAME=c) for template in CAMERA_TRAP_TEMPLATE]
    else:
        texts = [template.format(CLZ_NAME=c) for template in BIOCLIP_TEMPLATE]
    return texts


def get_class_embedding(model, tokenizer, embed_dim, class_name_idx, text_template='openai'): 
    device = next(model.parameters()).device
    context_length = model.context_length
    with torch.no_grad():
        class_embedding = torch.empty(len(class_name_idx), embed_dim)
        for class_name, class_idx in class_name_idx.items():
            # logging.info(f'Getting class embedding for {class_name}... ')
            texts = get_texts(class_name, text_template)
            # logging.info('Texts: ')
            # for t in texts:
            #     logging.info(f'\t{t}')
            texts = tokenizer(
                texts, 
                padding='max_length', 
                truncation=True, 
                max_length=context_length, 
                return_tensors='pt'
            )
            input_ids = texts['input_ids'].to(device)
            _class_embedding = model.encode_text(input_ids)
            _class_embedding = F.normalize(_class_embedding, dim=-1).mean(dim=0)
            _class_embedding = F.normalize(_class_embedding, dim=-1)
            class_embedding[class_idx] = _class_embedding
    return class_embedding

def get_model(params, class_num, text_model=None):
    # if torch.cuda.is_available():
    #     params.device = torch.cuda.current_device()
    # else:
    #     raise Exception("No GPU available")

    model = get_base_model(params, class_num)

    ##########
    tune_parameters = []
    if params.debug:
        logging.info("Trainable params:")

    if params.bitfit or params.difffit:
        TUNE_MODULES.append('bias')

    if params.ln or params.difffit:
        TUNE_MODULES.append('norm')

    if params.mlp_index:
        if isinstance(params.mlp_index, str):
            params.mlp_index = eval(params.mlp_index)
        for i in params.mlp_index:
            if params.mlp_type == 'fc1':
                TUNE_MODULES.append(str(i) + '.mlp.fc1')
            elif params.mlp_type == 'fc2':
                TUNE_MODULES.append(str(i) + '.mlp.fc2')
            elif params.mlp_type == 'full':
                TUNE_MODULES.append(str(i) + '.mlp.fc1')
                TUNE_MODULES.append(str(i) + '.mlp.fc2')
            else:
                raise NotImplementedError

    if params.attention_index:
        if isinstance(params.attention_index, str):
            params.attention_index = eval(params.attention_index)
        for i in params.attention_index:
            if params.attention_type == 'qkv':
                TUNE_MODULES.append(str(i) + '.attn.qkv')
            elif params.attention_type == 'proj':
                TUNE_MODULES.append(str(i) + '.attn.proj')
            elif params.attention_type == 'full':
                TUNE_MODULES.append(str(i) + '.attn.qkv')
                TUNE_MODULES.append(str(i) + '.attn.proj')
            else:
                raise NotImplementedError

    if params.block_index:
        if isinstance(params.block_index, str):
            params.block_index = eval(params.block_index)
        for i in params.block_index:
            TUNE_MODULES.append('blocks.' + str(i))

    for name, parameter in model.named_parameters():
        if params.full:
            parameter.requires_grad = True
            tune_parameters.append(parameter)
            if params.debug:
                logging.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
        else:
            if any(m in name for m in TUNE_MODULES):
                parameter.requires_grad = True
                tune_parameters.append(parameter)
                if params.debug:
                    logging.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
            else:
                parameter.requires_grad = False
    
    for name, parameter in text_model.named_parameters():
        if 'visual' in name:
            continue
        if params.text == 'head':
            parameter.requires_grad = False
        elif params.text == 'full':
            parameter.requires_grad = True
            if params.debug:
                logging.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
        elif params.text == 'lora':
            if 'lora' in name:
                parameter.requires_grad = True
                tune_parameters.append(parameter)
                if params.debug:
                    logging.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
            else:
                parameter.requires_grad = False
        else:
            raise NotImplementedError(f"Not implemented yet: {params.text}")

    train_text = True if params.text != 'head' else False
    model_grad_params_no_head = log_model_info(model, text_model, train_text)

    model = model.cuda(device=params.device)
    return model, tune_parameters, model_grad_params_no_head

def get_base_model(params, class_num):
    if params.pretrained_weights == "vit_base_patch16_224_in21k":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_224_in21k_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False, params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_16_in21k.npz', model_type='clip')
        model.reset_classifier(class_num)
    elif params.pretrained_weights == "vit_base_mae":
        model = timm.create_model("vit_base_patch16_224_in21k_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/mae_pretrain_vit_base.pth', model_type='clip')
        model.reset_classifier(class_num)
    elif params.pretrained_weights == "vit_base_patch14_dinov2":
        params.patch_size = 14
        model = timm.create_model("vit_base_patch14_dinov2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_14_dinov2.pth', model_type='dinov2')
        model.reset_classifier(class_num)
    elif params.pretrained_weights == 'vit_base_patch16_clip_224':
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_clip_224_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_16_clip.bin', model_type='clip')
        model.reset_classifier(class_num)
    ## Bioclip, we can tried other models as well
    elif params.pretrained_weights == 'bioclip':
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_clip_224_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False, params=params)
        model.load_pretrained(
            'pretrained_weights/bioclip/open_clip_pytorch_model.bin', model_type='bioclip')
        model.reset_classifier(class_num)
    elif params.pretrained_weights == 'bioclip2':
        params.patch_size = 14
        model = timm.create_model("vit_large_patch14_clip_224_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False, params=params)
        model.load_pretrained(
            'pretrained_weights/bioclip-2/open_clip_pytorch_model.bin', model_type='bioclip2')
        model.reset_classifier(class_num)
    else:
        raise NotImplementedError
    return model

def log_model_info(model, text_model, train_text, verbose=False):
    """Logs model info"""
    if verbose:
        logging.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    text_model_total_params = sum(p.numel() for p in text_model.parameters()) if train_text else 0
    model_total_params = model_total_params + text_model_total_params
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    text_model_grad_params = sum(
        p.numel() for p in text_model.parameters() if p.requires_grad) if train_text else 0
    model_grad_params = model_grad_params + text_model_grad_params
    model_grad_params_no_head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'head' not in n)
    logging.info("Total Parameters: {0}\t Gradient Parameters: {1}\t Gradient Parameters No Head: {2}".format(
        model_total_params, model_grad_params, model_grad_params_no_head))
    logging.info(f"total tuned percent:{(model_grad_params/model_total_params*100):.2f} %")
    logging.info(f"total tuned percent no head:{(model_grad_params_no_head / model_total_params * 100):.2f} %")
    ## Freeze the head
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = False
    return model_grad_params_no_head