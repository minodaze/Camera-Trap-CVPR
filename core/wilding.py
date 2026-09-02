"""
WildIng: Wildlife image Invariant representation model for geographical domain shift.
Implements the architecture from Santamaria et al. (2026), arXiv:2601.00993.

Three branches (Section 3.2):
  T  – class text centroids from LLM descriptions       (frozen)
  V  – image embeddings from visual encoder              (frozen by default)
  L  – image-text embeddings: VLM desc → text enc → MLP (trainable MLP only)

Similarity (Eq. 7): S = alpha * cosine(V, T) + (1 - alpha) * cosine(L, T)
Loss      (Eq. 8): cross_entropy(S / tau, labels)
"""

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class WildIngMLP(nn.Module):
    """
    Trainable MLP for the image-text branch (Section 3.5).
    Single hidden layer + ReLU + skip connection, matching the paper's
    MLP architecture used to bridge the VLM and CLIP embedding spaces.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Skip connection: linear projection when dims differ, identity otherwise
        self.skip = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x))) + self.skip(x)


class WildIngClassifier(nn.Module):
    """
    Wraps a frozen visual encoder with the full WildIng recipe.

    Args:
        visual_model:     OpenCLIP / BioCLIP visual encoder (nn.Module, frozen by default)
        mlp:              WildIngMLP  –  the only trainable part during out-of-domain training
        class_centroids:  [num_classes, F]  pre-computed, L2-normalized class text centroids
        alpha:            weight for image-only similarity branch (W); 1-alpha for image-text (Q)
        tau:              temperature for the contrastive loss
    """

    def __init__(
        self,
        visual_model: nn.Module,
        mlp: WildIngMLP,
        class_centroids: torch.Tensor,
        alpha: float = 0.5,
        tau: float = 0.1,
    ):
        super().__init__()
        self.visual_model = visual_model
        self.mlp = mlp
        # Stored as a buffer so it moves to the right device automatically
        self.register_buffer("class_centroids", F.normalize(class_centroids.float(), dim=-1))
        self.alpha = alpha
        self.tau = tau

    @property
    def embed_dim(self) -> int:
        return self.class_centroids.size(1)

    # ------------------------------------------------------------------
    # Compatibility shims expected by the rest of the codebase
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return True

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Normalized image features, same interface as CLIPClassifier."""
        return F.normalize(self.visual_model(images), dim=-1)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        desc_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images:    [B, C, H, W]
            desc_embs: [B, D]  pre-encoded VLM description embeddings, or None
                       When None the model falls back to the image-only branch (W only),
                       which is useful for evaluation without a desc cache.
        Returns:
            logits:    [B, num_classes]  (similarity scores already scaled by 1/tau)
        """
        V = self.forward_features(images)        # [B, F]
        T = self.class_centroids                 # [num_classes, F]  (already normalised)
        W = V @ T.T                              # [B, num_classes]

        if desc_embs is not None:
            L = F.normalize(self.mlp(desc_embs.to(V)), dim=-1)  # [B, F]
            Q = L @ T.T                          # [B, num_classes]
            S = self.alpha * W + (1.0 - self.alpha) * Q
        else:
            S = W

        return S / self.tau

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        logging.info(f"WildIngClassifier saved to {path}")

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
        logging.info(f"WildIngClassifier loaded from {path}")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def wilding_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    WildIng contrastive loss (Eq. 8).  Since forward() already divides by tau,
    this is just standard cross-entropy over the class dimension.
    """
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Helpers: building class centroids from precomputed descriptions
# ---------------------------------------------------------------------------

def build_class_centroids(
    desc_json: dict,          # {class_name: [str, str, ...]}  LLM descriptions
    text_encoder,             # callable: list[str] -> Tensor[N, F]
    class_name_idx: dict,     # {class_name: int}
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode LLM-generated descriptions with the text encoder, average per class,
    and return the L2-normalised centroid matrix T ∈ R^{|C| × F}.

    Args:
        desc_json:      mapping class_name → list of description strings
        text_encoder:   function (list[str]) → Tensor[N, F]  (already on device)
        class_name_idx: mapping class_name → class index
        embed_dim:      embedding dimension F
        device:         target device
    """
    centroids = torch.zeros(len(class_name_idx), embed_dim, device=device)
    with torch.no_grad():
        for class_name, class_idx in class_name_idx.items():
            descs = desc_json.get(class_name, [f"a photo of {class_name}."])
            if not descs:
                descs = [f"a photo of {class_name}."]
            embs = text_encoder(descs)              # [M_c, F]
            embs = F.normalize(embs.float(), dim=-1)
            centroid = F.normalize(embs.mean(dim=0), dim=-1)
            centroids[class_idx] = centroid
    return centroids


def load_class_centroids(
    centroids_path: str,
    class_name_idx: dict,
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load pre-computed centroid file produced by scripts/precompute_class_descriptions.py.
    The file is a dict {class_name: Tensor[F]} saved with torch.save.
    """
    data = torch.load(centroids_path, map_location="cpu")
    centroids = torch.zeros(len(class_name_idx), embed_dim)
    missing = []
    for class_name, class_idx in class_name_idx.items():
        if class_name in data:
            centroids[class_idx] = data[class_name].float()
        else:
            missing.append(class_name)
    if missing:
        logging.warning(
            f"WildIng: {len(missing)} classes not found in centroids file, "
            f"using zero vectors: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return F.normalize(centroids, dim=-1).to(device)


def load_desc_cache(desc_cache_path: str, device: torch.device = torch.device("cpu")) -> dict:
    """
    Load per-image VLM description embedding cache produced by
    scripts/precompute_vlm_descriptions.py.
    Returns a dict {file_path: Tensor[F]} kept on CPU for low memory usage.
    """
    cache = torch.load(desc_cache_path, map_location="cpu")
    logging.info(f"WildIng: loaded desc cache with {len(cache)} entries from {desc_cache_path}")
    return cache


def lookup_desc_embs(
    file_paths: list,
    desc_cache: dict,
    embed_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Build a [B, F] tensor of description embeddings by looking up file_paths in
    desc_cache.  Returns None if the cache is empty/None or if every path in the
    batch misses the cache — the caller should fall back to the image-only W branch.

    Passing zeros for missing entries is unsafe: MLP(0) becomes a class-frequency
    bias that can hurt val/test performance as training progresses.
    """
    if not desc_cache:
        return None
    embs = []
    hit = []
    for fp in file_paths:
        if fp in desc_cache:
            embs.append(desc_cache[fp])
            hit.append(True)
        else:
            embs.append(torch.zeros(embed_dim))
            hit.append(False)
    # If no image in this batch has a cached description, fall back to W-only.
    if not any(hit):
        return None
    return torch.stack(embs, dim=0).to(device)
