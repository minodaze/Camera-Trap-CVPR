"""
Precompute BioCLIP-2 text-encoder embeddings for GPT class descriptions
and save per-class centroid vectors.

Reads:  config/LLM_description/species_descriptions.json
        (dict {species_name: [desc1, desc2, ...]} produced by scripts/gpt_prompt.py)

Writes: config/LLM_description/class_centroids.pt
        (dict {class_name: Tensor[F]} on CPU, ready for build_wilding_classifier)

Run once before WildIng training:
    python scripts/precompute_class_descriptions.py
    python scripts/precompute_class_descriptions.py --pretrained-weights bioclip2
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from core.open_clip import create_model_and_transforms, get_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--desc-json",
        default="config/LLM_description/species_descriptions.json",
        help="GPT class descriptions JSON produced by scripts/gpt_prompt.py",
    )
    p.add_argument(
        "--output",
        default="config/LLM_description/class_centroids.pt",
        help="Output .pt file path ({class_name: Tensor[F]})",
    )
    p.add_argument(
        "--pretrained-weights",
        default="bioclip2",
        choices=["bioclip", "bioclip2"],
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_text_encoder(pretrained_weights: str, device: torch.device):
    """Load BioCLIP / BioCLIP-2 text encoder from local weights."""
    if pretrained_weights == "bioclip":
        weight_path = "pretrained_weights/bioclip/open_clip_pytorch_model.bin"
        model, _, _ = create_model_and_transforms(
            "ViT-B-16", weight_path,
            precision="amp", device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=None, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            aug_cfg={}, output_dict=True,
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("pretrained_weights/bioclip")
    else:  # bioclip2
        weight_path = "pretrained_weights/bioclip-2/open_clip_pytorch_model.bin"
        model, _, _ = create_model_and_transforms(
            "ViT-L-14", weight_path,
            precision="amp", device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=None, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            aug_cfg={}, output_dict=True,
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("pretrained_weights/bioclip-2")

    model = model.to(device).eval()
    context_length = model.context_length
    embed_dim = model.embed_dim

    @torch.no_grad()
    def encode(texts: list) -> torch.Tensor:
        """Encode a list of strings → normalized [N, F] Tensor on CPU."""
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        embs = model.encode_text(input_ids)
        return F.normalize(embs.float(), dim=-1).cpu()

    return encode, embed_dim


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.pretrained_weights} text encoder on {device}...")
    encode, embed_dim = load_text_encoder(args.pretrained_weights, device)
    print(f"Embedding dim: {embed_dim}")

    print(f"Reading GPT descriptions from {args.desc_json}...")
    with open(args.desc_json, "r") as f:
        desc_json = json.load(f)

    print(f"Computing centroids for {len(desc_json)} classes...")
    centroids = {}
    for class_name, descs in desc_json.items():
        if not descs:
            descs = [f"a photo of {class_name}."]

        # Encode in batches
        all_embs = []
        for i in range(0, len(descs), args.batch_size):
            batch = descs[i : i + args.batch_size]
            embs = encode(batch)   # [B, F]
            all_embs.append(embs)
        all_embs = torch.cat(all_embs, dim=0)  # [M_c, F]

        centroid = F.normalize(all_embs.mean(dim=0), dim=-1)  # [F]
        centroids[class_name] = centroid

        print(f"  {class_name}: {len(descs)} descriptions → centroid {centroid.shape}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(centroids, args.output)
    print(f"\nSaved {len(centroids)} class centroids → {args.output}")
    print(f"Centroid shape: {next(iter(centroids.values())).shape}")


if __name__ == "__main__":
    main()
