"""
WildIngDataset: Dataset for training the WildIng MLP.

Each sample provides:
  image       – PIL image loaded from img_path, transformed for BioCLIP
  desc_emb    – [F] pre-encoded VLM description embedding (from desc_cache)
  label       – integer class index

The VLM descriptions come from scripts/vlm_img.py output
  (config/LLM_description/species_vlm_descriptions.json).

Embed the raw descriptions ONCE with scripts/precompute_vlm_descriptions.py
and pass the resulting .pt cache here via `desc_cache`.
This avoids re-running the text encoder on every training step.
"""

import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class WildIngDataset(Dataset):
    """
    Dataset that returns (image, desc_emb, label) triples for WildIng training.

    Args:
        vlm_json_path:   path to species_vlm_descriptions.json produced by vlm_img.py
                         Format: list of {img_path, species, description}
        class_name_idx:  dict mapping species name → integer class index
        transform:       image transform (use BioCLIP's preprocess_train)
        desc_cache:      dict {img_path: Tensor[F]} of pre-encoded VLM embeddings.
                         If None, desc_emb will be returned as None and the model
                         falls back to image-only mode (alpha=1 effectively).
        embed_dim:       embedding dimension F (used to return zero vector on cache miss)
    """

    def __init__(
        self,
        vlm_json_path: str,
        class_name_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        desc_cache: Optional[Dict[str, torch.Tensor]] = None,
        embed_dim: int = 512,
    ):
        with open(vlm_json_path, "r") as f:
            raw = json.load(f)

        self.transform = transform
        self.class_name_idx = class_name_idx
        self.desc_cache = desc_cache or {}
        self.embed_dim = embed_dim

        # Filter to samples whose species is in the class index
        self.samples: List[Tuple[str, str, int]] = []  # (img_path, species, label)
        skipped = 0
        for entry in raw:
            species = entry["species"]
            if species not in class_name_idx:
                skipped += 1
                continue
            self.samples.append(
                (entry["img_path"], species, class_name_idx[species])
            )
        if skipped:
            log.warning(
                f"WildIngDataset: skipped {skipped} entries with unknown species."
            )
        log.info(
            f"WildIngDataset: {len(self.samples)} samples, "
            f"{len(class_name_idx)} classes."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, species, label = self.samples[idx]

        # ----- image -----
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # ----- VLM description embedding -----
        if self.desc_cache:
            desc_emb = self.desc_cache.get(img_path)
            if desc_emb is None:
                log.debug(f"Cache miss for {img_path}, using zero vector.")
                desc_emb = torch.zeros(self.embed_dim)
            else:
                desc_emb = desc_emb.float()
        else:
            desc_emb = torch.zeros(self.embed_dim)

        return image, desc_emb, label


def build_class_name_idx(vlm_json_path: str) -> Dict[str, int]:
    """
    Build a {species: int} mapping from the VLM JSON, sorted alphabetically
    so the index is deterministic across runs.
    """
    with open(vlm_json_path, "r") as f:
        raw = json.load(f)
    species_set = sorted({entry["species"] for entry in raw})
    return {s: i for i, s in enumerate(species_set)}


def build_loaders(
    vlm_json_path: str,
    class_name_idx: Dict[str, int],
    transform_train: Callable,
    transform_val: Optional[Callable],
    desc_cache: Optional[Dict[str, torch.Tensor]],
    embed_dim: int = 512,
    val_split: float = 0.15,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split VLM data into train/val and return DataLoaders.

    Args:
        val_split:  fraction of samples held out for validation
    """
    full_ds = WildIngDataset(
        vlm_json_path=vlm_json_path,
        class_name_idx=class_name_idx,
        transform=transform_train,
        desc_cache=desc_cache,
        embed_dim=embed_dim,
    )

    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=gen)

    # Val split uses val transform if provided
    if transform_val is not None:
        val_ds.dataset = WildIngDataset(
            vlm_json_path=vlm_json_path,
            class_name_idx=class_name_idx,
            transform=transform_val,
            desc_cache=desc_cache,
            embed_dim=embed_dim,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
