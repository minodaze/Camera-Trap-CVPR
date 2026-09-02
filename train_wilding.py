"""
WildIng accumulative-scratch continual learning training script.

Implements the architecture from Santamaria et al. (arXiv:2601.00993) and
runs it as accumulative-scratch CL: at each checkpoint, accumulate all data
seen so far, reset the MLP, and retrain from scratch.

Prerequisites (run once before this script):
    python scripts/vlm_img.py          ->  config/LLM_description/species_vlm_descriptions.json
    python scripts/gpt_prompt.py       ->  config/LLM_description/species_descriptions.json
    python scripts/precompute_vlm_descriptions.py  ->  config/LLM_description/vlm_desc_cache.pt

Usage:
    python train_wilding.py \\
        --train-json  /path/to/train.json \\
        --test-json   /path/to/test.json  \\
        --llm-json    config/LLM_description/species_descriptions.json \\
        --desc-cache  config/LLM_description/vlm_desc_cache.pt \\
        --output-dir  my_weights/wilding
"""

import argparse
import json
import logging
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
from copy import deepcopy

import numpy as np
import torch
import wandb
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core.data import CkpDataset
from core.open_clip import create_model_and_transforms, get_tokenizer
from core.wilding import (
    WildIngClassifier,
    WildIngMLP,
    build_class_centroids,
    load_desc_cache,
    lookup_desc_embs,
    wilding_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="WildIng accumulative-scratch CL training")
    # Data
    p.add_argument("--train-json", default="/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAME09/30/train.json",
                   help="Training JSON in CkpDataset format (pipeline's train.json)")
    p.add_argument("--test-json",   default="/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAME09/30/test.json", 
                   help="Test JSON in CkpDataset format (pipeline's test.json)")
    p.add_argument("--llm-json",    default="config/LLM_description/nz_EFH_HCAME09_species_descriptions.json", 
                   help="LLM class descriptions JSON (gpt_prompt.py output)")
    p.add_argument("--desc-cache",  default="config/LLM_description/nz_EFH_HCAME09_vlm_desc_cache.pt", 
                   help="Pre-encoded VLM embeddings .pt (precompute_vlm_descriptions.py output)")
    p.add_argument("--label-type",  default="common",
                   help="Field in JSON to use as class label (default: 'common')")
    p.add_argument("--output-dir",  default="/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/wilding/bioclip_nz_EFH_HCAME09",)
    p.add_argument("--pretrained-weights", default="bioclip2",
                   choices=["bioclip", "bioclip2", "longclip"])
    p.add_argument("--eval", action='store_true')
    p.add_argument("--log_dir", default="/fs/scratch/PAS2099/camera-trap-ECCV/ascend3/wilding/bioclip_nz_EFH_HCAME09_alpha0.6")

    # Hyperparameters (paper defaults, Section 4.2.1)
    p.add_argument("--hidden-dim",  type=int,   default=793,
                   help="MLP hidden layer dimension (paper default: 793)")
    p.add_argument("--alpha",       type=float, default=0.6,
                   help="Weight for image branch similarity alpha (paper default: 0.5)")
    p.add_argument("--tau",         type=float, default=0.1,
                   help="Temperature tau (paper default: 0.1)")
    p.add_argument("--epochs",      type=int,   default=30,
                   help="Training epochs per CL checkpoint (paper default: 30)")
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=0.09,
                   help="SGD learning rate (paper default: 0.09)")
    p.add_argument("--momentum",    type=float, default=0.80,
                   help="SGD momentum (paper default: 0.80)")
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cuda:1" if torch.cuda.is_available() else "cpu")

    # CL options
    p.add_argument("--start-ckp",   default=None,
                   help="Skip CL rounds before this checkpoint key (e.g. 'ckp_2')")
    p.add_argument("--early-stop",  type=int, default=5,
                   help="Stop if val bal_acc doesn't improve for this many epochs (0=disabled)")
    return p.parse_args()


# ── backbone loading ──────────────────────────────────────────────────────────

def load_backbone(pretrained_weights: str, device: torch.device):
    """Load BioCLIP / BioCLIP-2 from local pretrained_weights directory."""
    if pretrained_weights == "bioclip":
        log.info("Loading BioCLIP (ViT-B-16)...")
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-B-16",
            "pretrained_weights/bioclip/open_clip_pytorch_model.bin",
            precision="amp", device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=None, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            aug_cfg={}, output_dict=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("pretrained_weights/bioclip")
    elif pretrained_weights == "longclip":
        log.info("Loading Long-CLIP-B (ViT-B-16)...")
        model, preprocess = longclip.load('pretrained_weights/LongCLIP-B/longclip-B.pt', device=device)
        model = model.float()
        tokenizer = longclip.tokenize
        preprocess_train, preprocess_val = preprocess, preprocess
    else:
        log.info("Loading BioCLIP-2 (ViT-L-14)...")
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-L-14",
            "pretrained_weights/bioclip-2/open_clip_pytorch_model.bin",
            precision="amp", device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=None, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            aug_cfg={}, output_dict=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("pretrained_weights/bioclip-2")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, preprocess_train, preprocess_val, tokenizer


# ── text encoder wrapper (used only for building class centroids) ─────────────

def make_text_encoder(model, tokenizer, device):
    """Return a function: list[str] -> normalized Tensor[N, F] on CPU."""
    context_length = getattr(model, 'context_length', 77)
    log.info(f"context length = {context_length}")

    # HuggingFace tokenizers (bioclip, bioclip2) return BatchEncoding;
    # longclip.tokenize returns a Tensor directly.
    is_hf_tokenizer = hasattr(tokenizer, 'model_max_length')

    @torch.no_grad()
    def encode(texts: list) -> torch.Tensor:
        if is_hf_tokenizer:
            tokens = tokenizer(
                texts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=context_length,
            ).input_ids.to(device)
        else:
            tokens = tokenizer(texts).to(device)
        embs = model.encode_text(tokens)
        return F.normalize(embs.float(), dim=-1).cpu()

    return encode


# ── one-epoch training ────────────────────────────────────────────────────────

def train_one_epoch(classifier, loader, optimizer, desc_cache, embed_dim, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    classifier.mlp.train()
    classifier.visual_model.eval()  # visual encoder stays frozen

    total_loss, total_correct, total_n = 0.0, 0, 0

    for images, labels, file_paths, _old_logits, _is_buf in loader:
        images = images.to(device)
        labels = labels.to(device)

        desc_embs = lookup_desc_embs(list(file_paths), desc_cache, embed_dim, device)

        logits = classifier(images, desc_embs)
        loss   = wilding_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        total_n    += len(labels)

    return total_loss / total_n, total_correct / total_n


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(classifier, loader, desc_cache, embed_dim, device):
    """Evaluate. Returns (avg_loss, accuracy, balanced_accuracy)."""
    classifier.mlp.eval()
    classifier.visual_model.eval()

    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels, file_paths, _old_logits, _is_buf in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Falls back to visual-only branch when desc_embs is None
        desc_embs = lookup_desc_embs(list(file_paths), desc_cache, embed_dim, device)

        logits = classifier(images, desc_embs)
        loss   = wilding_loss(logits, labels)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * len(labels)
        total_n    += len(labels)

    acc     = sum(p == l for p, l in zip(all_preds, all_labels)) / total_n
    bal_acc = recall_score(all_labels, all_preds, labels=np.unique(all_labels), average='macro', zero_division=0)
    return total_loss / total_n, acc, bal_acc


# ── single-round training (one CL checkpoint) ────────────────────────────────

def train_one_round(classifier, train_dset, val_dset, desc_cache, embed_dim, args, device, round_name, round_idx):
    """Train for one accumulative-scratch CL round. Returns (classifier, best_bal_acc)."""
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(train_dset) > args.batch_size,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = SGD(classifier.mlp.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_bal_acc = -1.0
    best_state   = None
    no_improve   = 0
    early_stop_warmup = 15

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            classifier, train_loader, optimizer, desc_cache, embed_dim, device
        )
        val_loss, val_acc, val_bal_acc = evaluate(
            classifier, val_loader, desc_cache, embed_dim, device
        )
        scheduler.step()

        log.info(
            f"[{round_name}] Epoch {epoch:3d}/{args.epochs}  "
            f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={val_bal_acc:.3f}"
        )

        wandb.log({
            "round": round_idx,
            "epoch_in_round": epoch,
            f"{round_name}/train_loss": train_loss,
            f"{round_name}/train_acc":  train_acc,
            f"{round_name}/val_loss":   val_loss,
            f"{round_name}/val_acc":    val_acc,
            f"{round_name}/val_bal_acc": val_bal_acc,
        })

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            best_state   = deepcopy(classifier.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if args.early_stop > 0 and epoch >= early_stop_warmup and no_improve >= args.early_stop:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        classifier.load_state_dict(best_state)

    return classifier, best_bal_acc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # File logger
    if args.eval:
        fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    else:
        fh = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    # Wandb
    wandb.init(
        project="wilding",
        name=os.path.basename(os.path.normpath(args.output_dir)),
        dir=args.output_dir,
        config=vars(args),
    )

    # 1. Load frozen backbone
    model, _preprocess_train, _preprocess_val, tokenizer = load_backbone(
        args.pretrained_weights, device
    )
    if hasattr(model, 'embed_dim'):
        embed_dim = model.embed_dim
    elif hasattr(model, 'text_projection') and model.text_projection is not None:
        embed_dim = model.text_projection.shape[1]
    else:
        embed_dim = 512
    log.info(f"Backbone embed_dim: {embed_dim}")

    # 2. Extract class names from both train and test JSONs
    log.info("Extracting class names from JSON files...")
    class_names = []
    for json_path in [args.train_json, args.test_json]:
        with open(json_path, "r") as f:
            data = json.load(f)
        for ckp_key, entries in data.items():
            if not ckp_key.startswith("ckp_"):
                continue
            for entry in entries:
                name = entry.get(args.label_type)
                if name and name not in class_names:
                    class_names.append(name)
    log.info(f"  {len(class_names)} unique classes found.")

    # 3. Load datasets (CkpDataset handles the pipeline JSON format)
    log.info("Loading CkpDatasets...")
    train_dset = CkpDataset(args.train_json, class_names, is_train=True,  label_type=args.label_type)
    test_dset  = CkpDataset(args.test_json,  class_names, is_train=False, label_type=args.label_type)
    log.info(f"  Train checkpoints: {train_dset.get_ckp_list()}  ({len(train_dset)} samples)")
    log.info(f"  Test: {len(test_dset)} samples")

    # 4. Build class centroids T from LLM (GPT) descriptions
    log.info(f"Building class centroids from {args.llm_json}...")
    with open(args.llm_json, "r") as f:
        llm_desc = json.load(f)

    class_name_idx = train_dset.class_name_idx
    text_encoder   = make_text_encoder(model, tokenizer, device)
    class_centroids = build_class_centroids(
        desc_json      = llm_desc,
        text_encoder   = text_encoder,
        class_name_idx = class_name_idx,
        embed_dim      = embed_dim,
        device         = device,
    )
    log.info(f"  Centroids shape: {class_centroids.shape}")

    # 5. Load pre-encoded VLM description cache
    log.info(f"Loading VLM desc cache from {args.desc_cache}...")
    desc_cache = load_desc_cache(args.desc_cache, device=torch.device("cpu"))
    log.info(f"  Cache: {len(desc_cache)} entries.")

    # 6. Build the template WildIngClassifier (only MLP is trainable)
    mlp_init = WildIngMLP(
        input_dim  = embed_dim,
        hidden_dim = args.hidden_dim,
        output_dim = embed_dim,
    )
    _classifier = WildIngClassifier(
        visual_model    = model.visual,
        mlp             = mlp_init,
        class_centroids = class_centroids,
        alpha           = args.alpha,
        tau             = args.tau,
    ).to(device)

    n_params = sum(p.numel() for p in mlp_init.parameters())
    log.info(f"  MLP trainable parameters: {n_params:,}")

    # Save class index for reproducibility
    with open(os.path.join(args.output_dir, "class_name_idx.json"), "w") as f:
        json.dump(class_name_idx, f, indent=2)

    # 7. CL loop — forward evaluation protocol:
    #    ckp[0]  : no training, zero-shot eval on ckp[0] test data
    #    ckp[i>0]: train on ckp[0..i-1], early-stop on ckp[i-1] test, final eval on ckp[i] test
    full_ckp_list = train_dset.get_ckp_list()
    log.info(f"\nCheckpoints ({len(full_ckp_list)}): {full_ckp_list}")

    if args.start_ckp:
        if args.start_ckp not in full_ckp_list:
            raise ValueError(f"--start-ckp '{args.start_ckp}' not in: {full_ckp_list}")
        start_idx = full_ckp_list.index(args.start_ckp)
        log.info(f"Starting from {args.start_ckp} (index {start_idx})")
    else:
        start_idx = 0

    results = {}

    # ── Round 0: zero-shot evaluation on first ckpt's test data ─────────────────
    if start_idx == 0:
        ckp0 = full_ckp_list[0]
        log.info(f"\n{'='*60}\nRound 0 (zero-shot): {ckp0}\n{'='*60}")
        ckp0_test = test_dset.get_subset(is_train=False, ckp_list=[ckp0])
        zs_loader = DataLoader(
            ckp0_test,
            batch_size=args.batch_size * 4,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        _classifier.alpha = 1.0  # zero-shot: only image-branch similarity
        log.info(f"Setting alpha={_classifier.alpha} for zero-shot evaluation...")
        zs_loss, zs_acc, zs_bal_acc = evaluate(_classifier, zs_loader, desc_cache, embed_dim, device)
        log.info(f"[{ckp0}] Zero-shot  loss={zs_loss:.4f}  acc={zs_acc:.3f}  bal_acc={zs_bal_acc:.3f}")
        results[ckp0] = {"zero_shot_bal_acc": zs_bal_acc, "zero_shot_acc": zs_acc, "n_train": 0}
        wandb.log({"cl_round": 0, "checkpoint": ckp0, "test_bal_acc": zs_bal_acc, "test_acc": zs_acc, "n_train_accum": 0})
        ckpt_path = os.path.join(args.output_dir, f"{ckp0}_zero_shot.pt")
        _classifier.save(ckpt_path)
    
    _classifier.alpha = args.alpha  # restore alpha for training
    log.info(f"Setting alpha={_classifier.alpha} for training...")
    # ── Rounds 1..N ──────────────────────────────────────────────────────────────
    for round_idx in range(max(1, start_idx), len(full_ckp_list)):
        ckp_current = full_ckp_list[round_idx]
        ckp_prev    = full_ckp_list[round_idx - 1]
        past_ckps   = full_ckp_list[:round_idx]

        train_data = train_dset.get_subset(is_train=True,  ckp_list=past_ckps)
        val_data   = test_dset.get_subset( is_train=False, ckp_list=[ckp_prev])
        test_data  = test_dset.get_subset( is_train=False, ckp_list=[ckp_current])

        log.info(
            f"\n{'='*60}\n"
            f"Round {round_idx}/{len(full_ckp_list)-1}: "
            f"train={past_ckps}  val={ckp_prev}  test={ckp_current}  "
            f"({len(train_data)} train, {len(val_data)} val, {len(test_data)} test)\n"
            f"{'='*60}"
        )

        ckpt_path = os.path.join(args.output_dir, f"{ckp_current}_best.pt")

        classifier = deepcopy(_classifier)
        best_val_bal_acc = None
        if not os.path.exists(ckpt_path):
            log.info(f"No checkpoint from {ckpt_path}, start training")
            classifier, best_val_bal_acc = train_one_round(
                classifier  = classifier,
                train_dset  = train_data,
                val_dset    = val_data,
                desc_cache  = desc_cache,
                embed_dim   = embed_dim,
                args        = args,
                device      = device,
                round_name  = ckp_current,
                round_idx   = round_idx,
            )
            classifier.save(ckpt_path)
        else:
            log.info(f"Loading checkpoint from {ckpt_path}")
            classifier.load(ckpt_path)
            # Final test on current ckpt's unseen test data
        test_loader = DataLoader(
                test_data,
                batch_size=args.batch_size * 4,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        test_loss, test_acc, test_bal_acc = evaluate(classifier, test_loader, desc_cache, embed_dim, device)


        # log.info(
        #         f"[{ckp_current}]  val_bal_acc={best_val_bal_acc:.4f} (on {ckp_prev})  "
        #         f"test_bal_acc={test_bal_acc:.4f} test_acc={test_acc:.3f} (on {ckp_current})"
        #     )            

        results[ckp_current] = {
            "val_bal_acc":  best_val_bal_acc if best_val_bal_acc else 0.0,
            "test_bal_acc": test_bal_acc,
            "test_acc":     test_acc,
            "n_train":      len(train_data),
            "val_ckp":      ckp_prev,
            "test_ckp":     ckp_current,
        }
        wandb.log({
            "cl_round":      round_idx,
            "checkpoint":    ckp_current,
            "val_bal_acc":   best_val_bal_acc,
            "test_bal_acc":  test_bal_acc,
            "n_train_accum": len(train_data),
        })

    bal_accs = []
    for ckp, r in results.items():
        bal_accs.append(r.get("test_bal_acc", r.get("zero_shot_bal_acc", 0.0)))
    avg_bal_acc = sum(bal_accs) / len(bal_accs) if bal_accs else 0.0

    results["summary"] = {"avg_bal_acc": avg_bal_acc}
    # 8. Save summary
    if args.eval:
        summary_path = os.path.join(args.log_dir, "results.json")
    else:
        summary_path = os.path.join(args.output_dir, "results.json")
    
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n=== Final Results ===")
    for ckp, r in results.items():
        log.info(f"  {ckp}: {r}")

    results["summary"] = {"avg_bal_acc": avg_bal_acc, "n_ckpts": len(bal_accs)}
    log.info(f"  avg_bal_acc={avg_bal_acc:.4f} over {len(bal_accs)} checkpoints")
    wandb.log({"avg_bal_acc": avg_bal_acc})
    log.info(f"Summary -> {summary_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
