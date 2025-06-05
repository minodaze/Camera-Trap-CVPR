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

from .open_clip import create_model_and_transforms, get_cast_dtype, get_tokenizer


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


class CLIPClassifier(nn.Module):
    def __init__(self, visual_model, hidden_size):
        super(CLIPClassifier, self).__init__()
        self.visual_model = visual_model
        self.head = None
        self.initialized = False

    def init_head(self, class_embedding):
        assert not self.initialized, 'Head already initialized. '
        self.head = nn.Linear(class_embedding.size(1), class_embedding.size(0), bias=True)
        self.head.weight.data = class_embedding
        self.head.bias.data.zero_()
        self.initialized = True

    def reset_head(self):
        assert self.initialized, 'Head not initialized. '
        device = next(self.parameters()).device
        self.head = nn.Linear(self.head.in_features, self.head.out_features, bias=True).to(device)
    
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
        feats = F.normalize(x, dim=-1)
        x = self.head(feats)
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


def build_classifier(model_name, class_name_idx, device): 
    assert model_name in ['bioclip', 'open_clip']
    if isinstance(class_name_idx, list):
        class_name_idx = {c: i for i, c in enumerate(class_name_idx)}
    if model_name == 'bioclip':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-B-16',
            'pretrained_weight/bioclip/open_clip_pytorch_model.bin',
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
        tokenizer = AutoTokenizer.from_pretrained('pretrained_weight/bioclip')
        # img_size = model.visual.image_size
        classifier = CLIPClassifier(model.visual, model.embed_dim)
        class_embedding = get_class_embedding(model, tokenizer, class_name_idx)
        classifier.init_head(class_embedding)
        classifier = classifier.to(device)
    elif model_name == 'open_clip':
        assert False, 'Not implemented yet'
    return classifier


_LOOKUP_PATH = 'config/common_name_lookup.json'
_lookup = json.load(open(_LOOKUP_PATH))


def get_texts(c):
    use_bioclip_template = True
    if c not in _lookup:
        use_bioclip_template = False
    else:
        tax = _lookup[c]
        for t in tax:
            if not isinstance(t, str) and np.isnan(t):
                use_bioclip_template = False
                break
    if use_bioclip_template:
        tax = _lookup[c]
        common = c
        scientific = tax[-1]
        taxonomic = ' '.join(tax)
        scientific_common = f'{scientific} with common name {common}'
        taxonomic_common = f'{taxonomic} with common name {common}'
        names = [common, scientific, taxonomic, scientific_common, taxonomic_common]
        texts = []
        for n in names:
            texts += [template.format(CLZ_NAME=n) for template in BIOCLIP_TEMPLATE]
    else:
        texts = [template.format(CLZ_NAME=c) for template in OPENAI_IMAGENET_TEMPLATE]
    return texts


def get_class_embedding(model, tokenizer, class_name_idx): 
    device = next(model.parameters()).device
    context_length = model.context_length
    with torch.no_grad():
        class_embedding = torch.empty(len(class_name_idx), model.embed_dim)
        for class_name, class_idx in class_name_idx.items():
            # logging.info(f'Getting class embedding for {class_name}... ')
            texts = get_texts(class_name)
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
