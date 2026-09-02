"""
Precompute BioCLIP text-encoder embeddings for all VLM image descriptions.

Reads:  config/LLM_description/species_vlm_descriptions.json
        (list of {img_path, species, description} produced by scripts/vlm_img.py)

Writes: config/LLM_description/vlm_desc_cache.pt
        (dict {img_path: Tensor[F]} on CPU, ready for WildIngDataset)

Run once before training:
    python scripts/precompute_vlm_descriptions.py
"""

import argparse
import json
import sys
import os

# 1. Get the absolute path to the directory where your script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

long_clip_path = os.path.join(script_dir, "Long-CLIP")

# 3. Add it to the system path so Python can look inside it
if long_clip_path not in sys.path:
    sys.path.append(long_clip_path)

# 4. Now you can safely import the model module!
from model import longclip

from pathlib import Path
from transformers import AutoTokenizer

import torch
import torch.nn.functional as F

# ── BioCLIP loader (mirrors build_wilding_classifier in core/model.py) ─────
from core.open_clip import create_model_and_transforms, get_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vlm-json",
        default="config/LLM_description/species_vlm_descriptions.json",
        help="VLM descriptions JSON produced by scripts/vlm_img.py",
    )
    p.add_argument(
        "--output",
        default="config/LLM_description/vlm_desc_cache_longclip.pt",
        help="Output .pt file path",
    )
    p.add_argument(
        "--pretrained-weights",
        default="longclip",
        choices=["bioclip", "bioclip2", "longclip"],
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_text_encoder(pretrained_weights: str, device: torch.device):
    """Load BioCLIP / BioCLIP2 text encoder and tokenizer (frozen)."""
    if pretrained_weights == "bioclip":
        model, _, _ = create_model_and_transforms(
            "hf-hub:imageomics/bioclip",
            pretrained="hf-hub:imageomics/bioclip",
        )
    elif pretrained_weights == "longclip":
        print("Loading Long-CLIP-B (ViT-B-16)...")
        model, preprocess = longclip.load('pretrained_weights/LongCLIP-B/longclip-B.pt', device=device)
        tokenizer = longclip.tokenize
    else:  # bioclip2
        model, _, _ = create_model_and_transforms(
            'ViT-L-14',
            'pretrained_weights/bioclip-2/open_clip_pytorch_model.bin',
            precision='amp', device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=None, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            aug_cfg={}, output_dict=True,
        )
    model = model.to(device).eval()
    context_length = model.context_length  # typically 77 for ViT-L-14

    if pretrained_weights == "bioclip2":
        # bioclip2 uses HuggingFace AutoTokenizer; encode_text expects input_ids tensor
        tokenizer = AutoTokenizer.from_pretrained("pretrained_weights/bioclip-2")

        @torch.no_grad()
        def encode(texts: list) -> torch.Tensor:
            tokens = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            embs = model.encode_text(input_ids)
            return F.normalize(embs.float(), dim=-1).cpu()
    else:
        # Both BioCLIP and Long-CLIP tokenizers return plain tensors directly
        @torch.no_grad()
        def encode(texts: list) -> torch.Tensor:
            # longclip.tokenize natively handles the 248 padding internally
            tokens = tokenizer(texts).to(device)
            embs = model.encode_text(tokens)
            return F.normalize(embs.float(), dim=-1).cpu()

    if hasattr(model, 'embed_dim'):
        embed_dim = model.embed_dim
    elif hasattr(model, 'text_projection') and model.text_projection is not None:
        embed_dim = model.text_projection.shape[1]
    else:
        # Fallback for some ViT architectures if text_projection is abstracted differently
        embed_dim = 512
    return encode, embed_dim


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.pretrained_weights} text encoder on {device}...")
    encode, embed_dim = load_text_encoder(args.pretrained_weights, device)
    print(f"Embedding dim: {embed_dim}")

    print(f"Reading VLM descriptions from {args.vlm_json}...")
    with open(args.vlm_json, "r") as f:
        raw = json.load(f)

    # Build (img_path, description) pairs; skip entries with empty description
    pairs = []
    for entry in raw:
        desc = entry.get("description", "").strip()
        if desc:
            pairs.append((entry["img_path"], desc))
        else:
            print(f"  [skip] empty description for {entry['img_path']}")

    print(f"Encoding {len(pairs)} descriptions in batches of {args.batch_size}...")
    cache = {}
    for i in range(0, len(pairs), args.batch_size):
        batch = pairs[i : i + args.batch_size]
        paths, texts = zip(*batch)
        embs = encode(list(texts))  # [B, F]
        for path, emb in zip(paths, embs):
            cache[path] = emb
        if (i // args.batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(pairs)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(cache, args.output)
    print(f"Saved {len(cache)} embeddings → {args.output}")
    print(f"Embedding shape: {next(iter(cache.values())).shape}")


if __name__ == "__main__":
    main()
