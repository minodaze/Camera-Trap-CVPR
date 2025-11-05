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
        texts = [template.format(CLZ_NAME=c) for template in OPENAI_IMAGENET_TEMPLATE]
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

    def interpolate_lora(self, alpha=0.5):
        for block in self.visual_model.transformer.resblocks:
            block.attn.lora.params.merge_factor = alpha

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
        texts = [template.format(CLZ_NAME=c) for template in OPENAI_IMAGENET_TEMPLATE]
        return texts

def build_classifier(params, class_name_idx, device): 
    if isinstance(class_name_idx, list):
        class_name_idx = {c: i for i, c in enumerate(class_name_idx)}
    class_num = len(class_name_idx)
        
    # Log initial GPU memory before model loading
    if hasattr(params, 'gpu_memory_monitor') and params.gpu_memory_monitor:
        from utils.gpu_monitor import log_gpu_memory
        log_gpu_memory("model_build", "before_bioclip_load", device=device, enable_wandb=getattr(params, 'wandb', False))
    
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
    # use_bioclip_template = True
    # if c not in _lookup:
    #     use_bioclip_template = False
    # else:
    #     tax = _lookup[c]
    #     for t in tax:
    #         if not isinstance(t, str) and np.isnan(t):
    #             use_bioclip_template = False
    #             break
    # if use_bioclip_template and text_template == 'bioclip':
    #     tax = _lookup[c]
    #     common = c
    #     scientific = tax[-1]
    #     taxonomic = ' '.join(tax)
    #     scientific_common = f'{scientific} with common name {common}'
    #     taxonomic_common = f'{taxonomic} with common name {common}'
    #     names = [common, scientific, taxonomic, scientific_common, taxonomic_common]
    #     texts = []
    #     for n in names:
    #         texts += [template.format(CLZ_NAME=n) for template in BIOCLIP_TEMPLATE]
    # else:
    if text_template == 'customized':
        texts = [template.format(CLZ_NAME=c) for template in CAMERA_TRAP_TEMPLATE]
    else:
        texts = [template.format(CLZ_NAME=c) for template in OPENAI_IMAGENET_TEMPLATE]
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