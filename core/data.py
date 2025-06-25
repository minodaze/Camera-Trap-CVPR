import json
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, RandomHorizontalFlip, RandomResizedCrop
from collections import defaultdict
import logging
import copy
# import datetime
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

_IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


_TRAIN_TRANSFORM = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=_IMAGENET_DEFAULT_MEAN, std=_IMAGENET_DEFAULT_STD),
])
_CROP_TRAIN_TRANSFORM = Compose([
    RandomResizedCrop(224,                     # final H × W
                      scale=(0.7, 1.0),        # crop covers 70 – 100 % of image area
                      ratio=(3/4, 4/3),        # aspect-ratio range
                      interpolation=InterpolationMode.BICUBIC),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=_IMAGENET_DEFAULT_MEAN,
              std=_IMAGENET_DEFAULT_STD),
])

_VAL_TRANSFORM = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ToTensor(),
    Normalize(mean=_IMAGENET_DEFAULT_MEAN, std=_IMAGENET_DEFAULT_STD),
])

_VAL_TRANSFORM_SPECIESNET = Compose([
    Resize((480, 480), interpolation=InterpolationMode.BICUBIC),
    ToTensor(),1
])

@dataclass
class Sample:
    file_path: str
    label: int
    ckp: str
    timestamp: datetime
    logits: torch.Tensor = torch.empty(0)
    is_buf: bool = False

class SamplesDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = _VAL_TRANSFORM

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample.file_path
        label = sample.label
        logits = sample.logits
        is_buf = sample.is_buf
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)
        return image, label, logits, is_buf

class BufferDataset(SamplesDataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = _TRAIN_TRANSFORM

class ClassBalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.class_to_indices[sample.label].append(idx)
        self.classes = list(self.class_to_indices.keys())
        self.num_samples = len(self.dataset)

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples):
            selected_class = random.choice(self.classes)
            sample_idx = random.choice(self.class_to_indices[selected_class])
            indices.append(sample_idx)
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class CkpDataset(Dataset):
    _global_cache = {}

    def __init__(self, json_path, class_names, is_train=True, is_speciesnet=False, is_crop=False):
        self.cache = CkpDataset._global_cache
        self.json_path = json_path
        self.class_names = class_names
        self.is_train = is_train
        self.is_crop = is_crop
        self.crop_train_transform = _CROP_TRAIN_TRANSFORM
        self.train_transform = _TRAIN_TRANSFORM
        if is_speciesnet:
            self.val_transform = _VAL_TRANSFORM_SPECIESNET
        else:
            self.val_transform = _VAL_TRANSFORM
        if is_train:
            self.transform = self.crop_train_transform if self.is_crop else self.train_transform
        else:
            self.transform = self.val_transform 
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.class_name_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        self.samples = self._get_samples(data)
        class_num = len(self.class_names)
        for sample in self.samples:
            sample.logits = torch.empty(class_num)
        self.ckp_samples = self._get_ckp_samples(self.samples)
        self._sort_samples()

    def train(self):
        self.is_train = True
        self.transform = self.train_transform

    def eval(self):
        self.is_train = False
        self.transform = self.val_transform

    def _get_samples(self, data):
        samples = []
        for ckp, v in data.items():
            if not ckp.startswith("ckp_"):
                continue
            for data in v:
                file_path = data.get("image_path")
                if not file_path:
                    raise ValueError(f"image_path not found in {data}")
                class_name = data.get("class_name")
                if not class_name:
                    raise ValueError(f"class_name not found in {data}")
                assert class_name in self.class_names, f"class_name {class_name} not found in class_names. "
                label = self.class_name_idx.get(class_name)
                datetime_str = data.get("datetime")
                datetime_formats = [
                    "%Y:%m:%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y/%m/%dT%H:%M:%S",
                    "%Y%m%d %H:%M:%S",
                    "%Y%m%dT%H:%M:%S",
                ]
                for fmt in datetime_formats:
                    try:
                        timestamp = datetime.strptime(datetime_str, fmt)
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    raise ValueError(f"Unrecognized datetime format: {datetime_str}")
                # samples.append((file_path, label, ckp, timestamp))
                samples.append(Sample(file_path, label, ckp, timestamp))
        return samples

    def _get_ckp_samples(self, samples):
        ckp_samples = defaultdict(list)
        for sample in samples:
            ckp_samples[sample.ckp].append(sample)
        return ckp_samples

    def _sort_samples(self):
        # Sort self.samples by timestamp
        self.samples = sorted(self.samples, key=lambda x: x.timestamp)
        # Within each ckp, sort samples by timestamp
        for ckp in self.ckp_samples:
            self.ckp_samples[ckp] = sorted(self.ckp_samples[ckp], key=lambda x: x.timestamp)
        # Sort ckp_samples by key
        def _sort_key(k: str) -> int:
            _, v = k.split('_')
            return int(v)
        self.ckp_samples = dict(sorted(self.ckp_samples.items(), key=lambda x: _sort_key(x[0])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        assert self.transform is not None, "transform is not set, please call train() or val() first"
        sample = self.samples[idx]
        file_path = sample.file_path
        label = sample.label
        logits = sample.logits
        is_buf = sample.is_buf
        if file_path in self.cache:
            image = self.cache[file_path]
        else:
            image = Image.open(file_path).convert("RGB")
            self.cache[file_path] = image
        if self.is_crop and self.is_train:
            image = [self.transform(image), self.transform(image)]
        else:
            image = self.transform(image)
        return image, label, logits, is_buf

    def get_ckp_list(self):
        ckp_keys = list(self.ckp_samples.keys())
        return ckp_keys

    def get_subset(self, is_train, ckp_list=None, percentage=None):
        filtered_samples = []
        if percentage is not None:
            logging.info(f"Gathering samples from percentage: {percentage}, ignoring ckp_list")
            assert type(percentage) == float and 0 <= percentage <= 1, "percentage should be a float between 0 and 1. "
            for _, samples in self.ckp_samples.items():
                filtered_samples.extend(copy.deepcopy(samples))
            n_samples = int(len(filtered_samples) * percentage)
            filtered_samples = filtered_samples[:n_samples]
        elif ckp_list is not None:
            if not type(ckp_list) == list:
                ckp_list = [ckp_list]
            logging.info(f"Gathering samples from checkpoints: {ckp_list}")
            for ckp in ckp_list:
                assert ckp in self.ckp_samples, f"Checkpoint {ckp} not found"
                _filtered_samples = copy.deepcopy(self.ckp_samples[ckp])
                filtered_samples.extend(_filtered_samples)
        else:
            # Empty subset
            pass
        sub_dataset = CkpDataset.__new__(CkpDataset)
        sub_dataset.is_crop = self.is_crop
        sub_dataset.is_train = is_train
        sub_dataset.class_names = self.class_names
        sub_dataset.class_name_idx = self.class_name_idx
        if is_train:
            sub_dataset.transform = self.crop_train_transform if self.is_crop else self.train_transform
        else:
            sub_dataset.transform = self.val_transform
        sub_dataset.val_transform = self.val_transform
        sub_dataset.train_transform = self.train_transform
        sub_dataset.crop_train_transform = self.crop_train_transform
        sub_dataset.samples = filtered_samples
        sub_dataset.cache = self.cache
        logging.info(f"Subset length: {len(sub_dataset)}")
        return sub_dataset
    
    def add_samples(self, samples):
        new_samples = copy.deepcopy(self.samples)
        new_samples.extend(samples)
        self.samples = new_samples
        self.ckp_samples = self._get_ckp_samples(self.samples)
        self._sort_samples()
        logging.info(f"Added samples, new length: {len(self.samples)}")
    
    def apply_mask(self, mask):
        assert len(mask) == len(self.samples), "mask length should be equal to the number of samples. "
        self.samples = [sample for sample, m in zip(self.samples, mask) if m]
        self.ckp_samples = self._get_ckp_samples(self.samples)
        self._sort_samples()
        logging.info(f"Applied mask, new length: {len(self.samples)}")
