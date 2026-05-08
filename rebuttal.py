#!/usr/bin/env python3
"""
rebuttal.py – Training configuration & parameter analysis for rebuttal questions.

Usage (same flags as run_pipeline.py):
    python rebuttal.py --c /path/to/config.yaml --pretrained_weights bioclip2 --full
    python rebuttal.py --c /path/to/config.yaml --lora_bottleneck 16

Prints a detailed parameter breakdown by module showing frozen / full-FT / LoRA /
adapter / VPT counts, plus an overall summary.

Add --detailed to also print a per-transformer-block table.
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from types import ModuleType

import ruamel.yaml as yaml
import torch
import torch.nn as nn

# ── compatibility stub for broken huggingface_hub / transformers installs ─────
# Some cluster environments have a version mismatch. We inject a minimal mock so
# the model can be built for analysis without needing a real tokenizer.
def _install_transformers_stub():
    """Install a minimal transformers stub into sys.modules."""
    class _FakeTokenizer:
        context_length = 77

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            inst = cls()
            # try loading the real tokenizer file if available
            tok_file = os.path.join(path, "tokenizer.json")
            if os.path.exists(tok_file):
                try:
                    from tokenizers import Tokenizer as _HFTok
                    inst._inner = _HFTok.from_file(tok_file)
                    return inst
                except Exception:
                    pass
            return inst

        def __call__(self, texts, padding=None, truncation=None, max_length=None, return_tensors=None):
            L = max_length or self.context_length
            if isinstance(texts, str):
                texts = [texts]
            ids = torch.zeros(len(texts), L, dtype=torch.long)
            if hasattr(self, "_inner"):
                for i, t in enumerate(texts):
                    enc = self._inner.encode(t)
                    tok_ids = enc.ids[:L]
                    ids[i, :len(tok_ids)] = torch.tensor(tok_ids)
            return {"input_ids": ids}

    stub = ModuleType("transformers")
    stub.AutoTokenizer = _FakeTokenizer
    # also stub sub-modules that transformers/__init__.py tries to import
    for sub in ["dependency_versions_check", "utils", "utils.versions",
                "utils.hub", "utils.logging", "modeling_utils"]:
        sys.modules[f"transformers.{sub}"] = ModuleType(f"transformers.{sub}")
    sys.modules["transformers"] = stub


try:
    import transformers as _tr_check
    _tr_check.AutoTokenizer  # probe: will fail if version mismatch
except Exception:
    _install_transformers_stub()


_PETL_NEW_SITE = "/users/PAS2099/maizheda/anaconda3/envs/petl_new/lib/python3.10/site-packages"

def _install_timm():
    """Add petl_new site-packages to sys.path for timm/torchvision/ftfy/regex."""
    if os.path.isdir(_PETL_NEW_SITE) and _PETL_NEW_SITE not in sys.path:
        # Append rather than prepend so base-env packages still take priority
        sys.path.append(_PETL_NEW_SITE)
    try:
        import timm
        from timm.layers import DropPath  # quick probe
        return  # real timm found
    except Exception:
        # remove the path we just added and fall through to stub
        if _PETL_NEW_SITE in sys.path:
            sys.path.remove(_PETL_NEW_SITE)

    # minimal stub fallback (only covers block.py / mlp.py needs for bioclip2)
    def _to2(x):
        from collections.abc import Iterable
        return tuple(x) if isinstance(x, Iterable) and not isinstance(x, str) else (x, x)

    class _DropPath(nn.Module):
        def __init__(self, drop_prob=0., scale_by_keep=True): super().__init__()
        def forward(self, x): return x

    class _LayerScale(nn.Module):
        def __init__(self, dim, init_values=1e-5, inplace=False):
            super().__init__()
            self.inplace = inplace
            self.gamma = nn.Parameter(torch.full((dim,), float(init_values)))
        def forward(self, x):
            return x.mul_(self.gamma) if self.inplace else x * self.gamma

    _noop = lambda *a, **kw: None
    timm_root   = ModuleType("timm")
    timm_layers = ModuleType("timm.layers")
    timm_helpers = ModuleType("timm.layers.helpers")
    timm_trace  = ModuleType("timm.layers.trace_utils")
    timm_models = ModuleType("timm.models")
    timm_vit    = ModuleType("timm.models.vision_transformer")
    timm_builder = ModuleType("timm.models._builder")
    timm_manip  = ModuleType("timm.models._manipulate")
    timm_reg    = ModuleType("timm.models._registry")

    # timm.layers attributes
    for attr in ("DropPath", "AttentionPoolLatent", "PatchDropout", "SwiGLUPacked"):
        setattr(timm_layers, attr, _DropPath)
    timm_layers.LayerScale        = _LayerScale
    timm_layers.RmsNorm           = nn.LayerNorm
    for attr in ("trunc_normal_", "lecun_normal_", "resample_patch_embed",
                 "resample_abs_pos_embed", "use_fused_attn"):
        setattr(timm_layers, attr, _noop)
    timm_layers.get_act_layer  = lambda *_, **__: nn.GELU
    timm_layers.get_norm_layer = lambda *_, **__: nn.LayerNorm
    timm_layers.LayerType      = type  # just a type alias
    timm_helpers.to_2tuple = _to2
    timm_layers.to_2tuple  = _to2
    timm_trace._assert     = lambda c, m="": None
    timm_vit.VisionTransformer = nn.Module
    timm_vit.LayerScale        = _LayerScale
    for attr in ("init_weights_vit_timm", "get_init_weights_vit",
                 "_load_weights", "checkpoint_filter_fn"):
        setattr(timm_vit, attr, _noop)
    timm_builder.build_model_with_cfg = _noop
    for attr in ("named_apply", "checkpoint_seq", "adapt_input_conv"):
        setattr(timm_manip, attr, _noop)
    for attr in ("generate_default_cfgs", "register_model", "register_model_deprecations"):
        setattr(timm_reg, attr, lambda f=None, *a, **kw: (f if f else lambda x: x))

    for key, mod in [
        ("timm", timm_root), ("timm.layers", timm_layers),
        ("timm.layers.helpers", timm_helpers), ("timm.layers.trace_utils", timm_trace),
        ("timm.models", timm_models), ("timm.models.vision_transformer", timm_vit),
        ("timm.models._builder", timm_builder), ("timm.models._manipulate", timm_manip),
        ("timm.models._registry", timm_reg),
    ]:
        sys.modules[key] = mod
    timm_root.layers = timm_layers
    timm_root.models = timm_models


# Clear any broken timm cached state before loading
for _k in list(sys.modules.keys()):
    if _k == "timm" or _k.startswith("timm."):
        del sys.modules[_k]
_install_timm()


def _install_ftfy_stub():
    ftfy_mod = ModuleType("ftfy")
    ftfy_mod.fix_text = lambda s, **kw: s
    sys.modules["ftfy"] = ftfy_mod

try:
    import ftfy as _ftfy_check
except Exception:
    _install_ftfy_stub()


def _install_regex_stub():
    import re as _re
    regex_mod = ModuleType("regex")
    # expose the common re attributes
    for _attr in dir(_re):
        setattr(regex_mod, _attr, getattr(_re, _attr))
    sys.modules["regex"] = regex_mod

try:
    import regex as _regex_check
except Exception:
    _install_regex_stub()


# torchvision stub (only imports are needed; transforms/ops are never called during analysis)
try:
    import torchvision as _tv_check
    _tv_check.transforms  # probe
except Exception:
    def _make_tv_stub():
        _tv       = ModuleType("torchvision")
        _tv_ds    = ModuleType("torchvision.datasets")
        _tv_io    = ModuleType("torchvision.io")
        _tv_models = ModuleType("torchvision.models")
        _tv_ops   = ModuleType("torchvision.ops")
        _tv_ops_misc = ModuleType("torchvision.ops.misc")
        _tv_ops_misc.FrozenBatchNorm2d = nn.BatchNorm2d
        class _Interp:  # InterpolationMode enum-like
            BICUBIC = 3; BILINEAR = 2; NEAREST = 0
        class _Transform(nn.Module):  # base class for all transforms
            def __init__(self, *_, **__): super().__init__()
            def forward(self, x): return x
        _tv_tr    = ModuleType("torchvision.transforms")
        for _tr_attr in ("Normalize", "Compose", "RandomResizedCrop", "ToTensor",
                         "Resize", "CenterCrop", "RandomHorizontalFlip",
                         "ColorJitter", "RandomGrayscale", "RandomApply",
                         "GaussianBlur", "ToTensor", "Lambda"):
            setattr(_tv_tr, _tr_attr, _Transform)
        _tv_tr.InterpolationMode = _Interp
        _tv_tr_fn = ModuleType("torchvision.transforms.functional")
        _tv_tr_fn.normalize        = lambda x, *a, **kw: x
        _tv_tr_fn.resize           = lambda x, *a, **kw: x
        _tv_tr_fn.center_crop      = lambda x, *a, **kw: x
        _tv_tr_fn.to_tensor        = lambda x, *a, **kw: x
        _tv_utils = ModuleType("torchvision.utils")
        _tv._meta_registrations = ModuleType("torchvision._meta_registrations")
        _tv.datasets  = _tv_ds
        _tv.io        = _tv_io
        _tv.models    = _tv_models
        _tv.ops       = _tv_ops
        _tv.transforms = _tv_tr
        _tv.utils     = _tv_utils
        _tv_ops.misc  = _tv_ops_misc
        for _k, _m in [
            ("torchvision", _tv),
            ("torchvision.datasets", _tv_ds),
            ("torchvision.io", _tv_io),
            ("torchvision.models", _tv_models),
            ("torchvision.ops", _tv_ops),
            ("torchvision.ops.misc", _tv_ops_misc),
            ("torchvision.transforms", _tv_tr),
            ("torchvision.transforms.functional", _tv_tr_fn),
            ("torchvision.utils", _tv_utils),
            ("torchvision._meta_registrations", _tv._meta_registrations),
        ]:
            sys.modules[_k] = _m
    _make_tv_stub()

# stub wandb / matplotlib (not needed for analysis)
for _dep in ["wandb", "matplotlib", "matplotlib.pyplot", "matplotlib.patches",
             "matplotlib.colors"]:
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except Exception:
            sys.modules[_dep] = ModuleType(_dep)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.model import build_classifier


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fmt(n: int, w: int = 14) -> str:
    return f"{n:>{w},}"


def _pct(num: int, total: int) -> str:
    if total == 0:
        return "  N/A "
    return f"{100 * num / total:5.1f}%"


# All known PETL types (order matters for display)
_TYPES = ["full_ft", "lora", "adapter", "vpt", "ssf", "fact", "vqt", "difffit", "frozen"]


def _param_type(name: str, param: nn.Parameter) -> str:
    """Classify a parameter into its PETL role."""
    if not param.requires_grad:
        return "frozen"
    n = name.lower()
    if "lora" in n:
        return "lora"
    if "ft_attn_module" in n or "ft_mlp_module" in n or ("adapter" in n and "repad" not in n):
        return "adapter"
    if "repadapter" in n:
        return "adapter"
    if "vpt" in n:
        return "vpt"
    if "ssf_scale" in n or "ssf_shift" in n:
        return "ssf"
    if "fact" in n:
        return "fact"
    if "vqt" in n:
        return "vqt"
    if "difffit" in n:
        return "difffit"
    return "full_ft"


def _module_group(name: str):
    """
    Map a parameter name to (top_module, sub_group).
    Returns (None, None) for parameters that should be skipped (e.g. shared).
    """
    parts = name.split(".")

    if parts[0] == "head":
        return ("Classification Head", "linear_head")

    if parts[0] == "proj_head":
        return ("Projection Head", "proj")

    if parts[0] == "visual_model":
        if len(parts) < 2:
            return ("Image Encoder", "other")
        sub = parts[1]

        if sub in ("conv1", "class_embedding", "positional_embedding", "ln_pre", "patchnorm_pre_ln"):
            return ("Image Encoder", "stem")
        if sub == "ln_post":
            return ("Image Encoder", "post_ln")
        if sub == "proj":
            return ("Image Encoder", "out_proj")
        if sub == "attn_pool":
            return ("Image Encoder", "attn_pool")

        if sub == "transformer" and len(parts) >= 4 and parts[2] == "resblocks":
            block_idx = parts[3]
            component = parts[4] if len(parts) > 4 else "other"
            n_lower = name.lower()
            if "lora" in n_lower:
                return ("Image Encoder", f"block_{block_idx}_lora")
            if component in ("ln_1", "ln_2", "ln_attn"):
                return ("Image Encoder", f"block_{block_idx}_ln")
            if component in ("ls_1", "ls_2"):
                return ("Image Encoder", f"block_{block_idx}_ls")
            if component == "attn":
                return ("Image Encoder", f"block_{block_idx}_attn")
            if component == "mlp":
                return ("Image Encoder", f"block_{block_idx}_mlp")
            if component in ("ft_attn_module", "ft_mlp_module"):
                return ("Image Encoder", f"block_{block_idx}_adapter")
            return ("Image Encoder", f"block_{block_idx}_other")

        return ("Image Encoder", f"other_{sub}")

    if parts[0] == "text_model":
        # bioclip_model is stored here when text != 'head'.
        # Skip the .visual subtree – it's the same object as visual_model.
        if len(parts) > 1 and parts[1] == "visual":
            return (None, None)
        sub = parts[1] if len(parts) > 1 else "other"
        label_map = {
            "token_embedding": "token_embed",
            "positional_embedding": "pos_embed",
            "transformer": "transformer",
            "ln_final": "ln_final",
            "text_projection": "text_proj",
            "logit_scale": "logit_scale",
        }
        return ("Text Encoder", label_map.get(sub, f"other_{sub}"))

    return ("Other", parts[0])


# ─── analysis ─────────────────────────────────────────────────────────────────

def _analyze(classifier: nn.Module):
    """
    Walk all parameters (deduplicating shared tensors) and collect
      data[top_module][sub_group][param_type] = num_params
    Returns (data, ordered list of (top, sub) pairs).
    """
    seen = set()
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    order = []

    for name, param in classifier.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)

        top, sub = _module_group(name)
        if top is None:
            continue

        ptype = _param_type(name, param)
        data[top][sub][ptype] += param.numel()

        key = (top, sub)
        if key not in order:
            order.append(key)

    return data, order


def _merge(type_dicts):
    """Merge a list of {ptype: count} dicts into one."""
    out = defaultdict(int)
    for d in type_dicts:
        for k, v in d.items():
            out[k] += v
    return out


# ─── printing ─────────────────────────────────────────────────────────────────

_COL = 46   # module name column width
_W   = 118  # total line width

_HEADER = (
    f"{'Module':<{_COL}} "
    f"{'Total':>12}  "
    f"{'Trainable':>13}  "
    f"{'Frozen':>12}  "
    f"{'Full FT':>12}  "
    f"{'LoRA':>10}  "
    f"{'Adapter':>10}  "
    f"{'VPT':>8}"
)


def _row(label: str, counts: dict) -> str:
    total     = sum(counts.get(t, 0) for t in _TYPES)
    trainable = sum(counts.get(t, 0) for t in _TYPES if t != "frozen")
    frozen    = counts.get("frozen", 0)
    full_ft   = counts.get("full_ft", 0)
    lora      = counts.get("lora", 0)
    adapter   = counts.get("adapter", 0)
    vpt       = counts.get("vpt", 0)

    return (
        f"{label:<{_COL}} "
        f"{_fmt(total, 12)}  "
        f"{_fmt(trainable, 10)} {_pct(trainable, total)}  "
        f"{_fmt(frozen, 10)} {_pct(frozen, total)}  "
        f"{_fmt(full_ft, 10)}  "
        f"{_fmt(lora, 8)}  "
        f"{_fmt(adapter, 8)}  "
        f"{_fmt(vpt, 6)}"
    )


def _top_agg(data, top):
    """Aggregate all sub-groups for a top-level module."""
    out = defaultdict(int)
    for sub_dict in data[top].values():
        for ptype, cnt in sub_dict.items():
            out[ptype] += cnt
    return out


def _block_component_agg(data, top, suffix):
    """Sum all block_N_<suffix> sub-groups for image encoder."""
    out = defaultdict(int)
    for sub, type_dict in data[top].items():
        if sub.startswith("block_") and sub.endswith(suffix):
            for ptype, cnt in type_dict.items():
                out[ptype] += cnt
    return out


def _count_blocks(data, top):
    idxs = set()
    for sub in data[top]:
        if sub.startswith("block_"):
            try:
                idxs.add(int(sub.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return len(idxs)


def print_report(classifier: nn.Module, args, detailed: bool = False):
    data, order = _analyze(classifier)

    # Grand totals
    grand = defaultdict(int)
    for top in data:
        for sub_dict in data[top].values():
            for ptype, cnt in sub_dict.items():
                grand[ptype] += cnt
    grand_total     = sum(grand.values())
    grand_trainable = sum(v for k, v in grand.items() if k != "frozen")
    grand_frozen    = grand.get("frozen", 0)

    # Ordered list of top-level modules
    seen_tops = []
    for top, _ in order:
        if top not in seen_tops:
            seen_tops.append(top)

    # ── Header ───────────────────────────────────────────────────────────────
    config_name  = os.path.basename(getattr(args, "c", "N/A"))
    backbone     = getattr(args, "pretrained_weights", "?")
    full_ft_flag = getattr(args, "full", False)
    lora_r       = getattr(args, "lora_bottleneck", 0)
    ft_attn      = getattr(args, "ft_attn_module", None)
    ft_mlp       = getattr(args, "ft_mlp_module", None)
    text_mode    = getattr(args, "text", "head")

    if full_ft_flag:
        petl_desc = "Full Fine-tuning (--full)"
    elif lora_r > 0:
        petl_desc = f"LoRA  r={lora_r}"
    elif ft_attn or ft_mlp:
        petl_desc = f"Adapter  attn={ft_attn or 'none'}  mlp={ft_mlp or 'none'}"
    else:
        petl_desc = "Selective / Frozen (no explicit PETL)"

    print()
    print("=" * _W)
    print("  PARAMETER ANALYSIS REPORT".center(_W))
    print("=" * _W)
    print(f"  Config    : {config_name}")
    print(f"  Backbone  : {backbone}")
    print(f"  PETL Mode : {petl_desc}")
    print(f"  Text Mode : {text_mode}  "
          f"({'linear head from frozen text encoder' if text_mode == 'head' else 'text encoder in computation graph'})")
    print()

    # ── Module table ─────────────────────────────────────────────────────────
    print(_HEADER)
    print("─" * _W)

    for top in seen_tops:
        agg = _top_agg(data, top)
        print(_row(top, agg))

        if top == "Image Encoder":
            n_blocks  = _count_blocks(data, top)
            stem      = data[top].get("stem", {})
            post_ln   = data[top].get("post_ln", {})
            out_proj  = data[top].get("out_proj", {})
            attn_pool = data[top].get("attn_pool", {})
            other     = _merge([v for k, v in data[top].items() if k.endswith("_other")])

            if stem:
                print(_row("  ├─ Stem  (patch embed / cls / pos / ln_pre)", stem))

            if n_blocks:
                attn_all    = _block_component_agg(data, top, "_attn")
                mlp_all     = _block_component_agg(data, top, "_mlp")
                ln_all      = _block_component_agg(data, top, "_ln")
                ls_all      = _block_component_agg(data, top, "_ls")
                lora_all    = _block_component_agg(data, top, "_lora")
                adapter_all = _block_component_agg(data, top, "_adapter")
                blk_total   = _merge([attn_all, mlp_all, ln_all, ls_all, lora_all, adapter_all])
                for sub, td in data[top].items():
                    if sub.startswith("block_") and not any(
                        sub.endswith(s) for s in ("_attn", "_mlp", "_ln", "_ls", "_lora", "_adapter", "_other")
                    ):
                        for ptype, cnt in td.items():
                            blk_total[ptype] += cnt

                print(_row(f"  ├─ Transformer Blocks  (×{n_blocks})", blk_total))
                if any(attn_all.values()):
                    print(_row( "  │    ├─ Self-Attention  (QKV + out_proj)", attn_all))
                if any(mlp_all.values()):
                    print(_row( "  │    ├─ MLP", mlp_all))
                if any(ln_all.values()):
                    print(_row( "  │    ├─ Layer Norms  (ln_1 + ln_2)", ln_all))
                if any(ls_all.values()):
                    print(_row( "  │    ├─ Layer Scale  (ls_1 + ls_2)", ls_all))
                if any(lora_all.values()):
                    print(_row( "  │    ├─ LoRA  (A + B matrices)", lora_all))
                if any(adapter_all.values()):
                    print(_row( "  │    ├─ Adapter Modules", adapter_all))

            if any(v for v in attn_pool.values() if v):
                print(_row("  ├─ Attentional Pooler", attn_pool))
            if post_ln:
                print(_row("  ├─ Post LayerNorm  (ln_post)", post_ln))
            if out_proj:
                print(_row("  └─ Output Projection  (proj)", out_proj))
            if any(other.values()):
                print(_row("  └─ Other", other))

        elif top == "Text Encoder":
            label_map = {
                "token_embed": "  ├─ Token Embedding",
                "pos_embed":   "  ├─ Positional Embedding",
                "transformer": "  ├─ Transformer Blocks",
                "ln_final":    "  ├─ Final LayerNorm",
                "text_proj":   "  ├─ Text Projection",
                "logit_scale": "  └─ Logit Scale",
            }
            for sub, type_dict in data[top].items():
                lbl = label_map.get(sub, f"  └─ {sub}")
                print(_row(lbl, type_dict))

        elif top == "Classification Head":
            hd = data[top].get("linear_head", {})
            if hd:
                print(_row("  └─ Linear (class_emb × hidden)", hd))

        elif top == "Projection Head":
            ph = data[top].get("proj", {})
            if ph:
                print(_row("  └─ Projection", ph))

        else:
            for sub, type_dict in data[top].items():
                print(_row(f"  └─ {sub}", type_dict))

        print("─" * _W)

    print(_row("TOTAL", grand))
    print("=" * _W)

    # ── Per-block detail (optional) ───────────────────────────────────────────
    if detailed and "Image Encoder" in data:
        print()
        print("━" * _W)
        print("  PER-BLOCK DETAIL  (Image Encoder)".center(_W))
        print("━" * _W)
        print(_HEADER)
        print("─" * _W)
        block_idxs = sorted(set(
            int(sub.split("_")[1])
            for sub in data["Image Encoder"]
            if sub.startswith("block_")
        ))
        for idx in block_idxs:
            blk_agg = defaultdict(int)
            for sub, type_dict in data["Image Encoder"].items():
                if sub.startswith(f"block_{idx}_"):
                    for ptype, cnt in type_dict.items():
                        blk_agg[ptype] += cnt
            print(_row(f"  Block {idx:>2}", blk_agg))
            for suffix, label in [
                ("_attn",    f"    ├─ Attention"),
                ("_mlp",     f"    ├─ MLP"),
                ("_ln",      f"    ├─ LayerNorms"),
                ("_ls",      f"    ├─ LayerScale"),
                ("_lora",    f"    ├─ LoRA"),
                ("_adapter", f"    └─ Adapter"),
            ]:
                td = data["Image Encoder"].get(f"block_{idx}{suffix}", {})
                if any(td.values()):
                    print(_row(label, td))
        print("=" * _W)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("━" * _W)
    print("  OVERALL SUMMARY".center(_W))
    print("━" * _W)
    print(f"  Total Parameters  : {grand_total:>15,}   (100.0%)")
    print(f"  Trainable         : {grand_trainable:>15,}   ({_pct(grand_trainable, grand_total)})")
    print(f"  Frozen            : {grand_frozen:>15,}   ({_pct(grand_frozen, grand_total)})")
    print()
    print("  Trainable Breakdown:")
    type_labels = [
        ("full_ft",  "Full Fine-tuning"),
        ("lora",     "LoRA"),
        ("adapter",  "Adapter"),
        ("vpt",      "VPT  (Visual Prompt Tuning)"),
        ("ssf",      "SSF"),
        ("fact",     "FacT"),
        ("vqt",      "VQT"),
        ("difffit",  "DiffFit"),
    ]
    for ptype, label in type_labels:
        cnt = grand.get(ptype, 0)
        if cnt > 0 or ptype in ("full_ft", "lora", "adapter"):
            print(f"    {label:<30} : {cnt:>15,}   ({_pct(cnt, grand_total)})")
    print()
    print(f"  Effective Tuning Ratio : {grand_trainable:,} / {grand_total:,} = "
          f"{100 * grand_trainable / grand_total:.2f}%")
    print("━" * _W)
    print()


# ─── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parameter analysis tool (rebuttal helper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--c", type=str, required=True, help="YAML config file (same as run_pipeline.py)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for model building (cpu is fine – no training needed)")
    p.add_argument("--detailed", action="store_true",
                   help="Also print a per-transformer-block breakdown table")

    # ── PETL / model flags (mirror run_pipeline.py) ──────────────────────────
    p.add_argument("--pretrained_weights", type=str, default="bioclip2")
    p.add_argument("--full", action="store_true")
    p.add_argument("--text", type=str, default="head")
    p.add_argument("--text_template", type=str, default="openai")
    p.add_argument("--lora_bottleneck", type=int, default=0)
    p.add_argument("--merge_factor", type=float, default=1.0)
    p.add_argument("--lora_interpolate", action="store_true")
    p.add_argument("--lora_alpha", type=float, default=0.5)
    p.add_argument("--ft_attn_module", default=None, choices=[None, "adapter", "convpass", "repadapter"])
    p.add_argument("--ft_mlp_module",  default=None, choices=[None, "adapter", "convpass", "repadapter"])
    p.add_argument("--ft_attn_mode", default="parallel")
    p.add_argument("--ft_mlp_mode",  default="parallel")
    p.add_argument("--ft_attn_ln",   default="before")
    p.add_argument("--ft_mlp_ln",    default="before")
    p.add_argument("--adapter_bottleneck", type=int, default=64)
    p.add_argument("--adapter_init",  type=str, default="lora_kaiming")
    p.add_argument("--adapter_scaler", type=float, default=0.1)
    p.add_argument("--convpass_bottleneck", type=int, default=8)
    p.add_argument("--convpass_init", type=str, default="lora_xavier")
    p.add_argument("--convpass_scaler", type=float, default=10.0)
    p.add_argument("--convpass_xavier_init", action="store_true")
    p.add_argument("--vpt_mode", type=str, default=None, choices=[None, "deep", "shallow"])
    p.add_argument("--vpt_num",  type=int, default=10)
    p.add_argument("--vpt_layer", type=int, default=None)
    p.add_argument("--vpt_dropout", type=float, default=0.1)
    p.add_argument("--ssf", action="store_true")
    p.add_argument("--fact_dim",    type=int, default=8)
    p.add_argument("--fact_type",   type=str, default=None, choices=[None, "tk", "tt"])
    p.add_argument("--fact_scaler", type=float, default=1.0)
    p.add_argument("--repadapter_bottleneck", type=int, default=8)
    p.add_argument("--repadapter_init",    type=str, default="lora_xavier")
    p.add_argument("--repadapter_scaler",  type=float, default=1.0)
    p.add_argument("--repadapter_group",   type=int, default=2)
    p.add_argument("--bitfit",   action="store_true")
    p.add_argument("--vqt_num",     type=int, default=0)
    p.add_argument("--vqt_dropout", type=float, default=0.1)
    p.add_argument("--mlp_index",   type=int, nargs="+", default=None)
    p.add_argument("--mlp_type",    type=str, default="full")
    p.add_argument("--attention_index", type=int, nargs="+", default=None)
    p.add_argument("--attention_type",  type=str, default="full")
    p.add_argument("--block_index", type=int, nargs="+", default=None)
    p.add_argument("--ln",      action="store_true")
    p.add_argument("--difffit", action="store_true")
    p.add_argument("--drop_path_rate", type=float, default=0.0)

    # ── misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--label_type",  type=str, default="common", choices=["scientific", "common"])
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--seed",        type=int, default=9527)
    p.add_argument("--gpu_memory_monitor", action="store_true")
    p.add_argument("--wandb",       action="store_true")

    # absorb any extra flags from run_pipeline.py that we don't need
    args, _ = p.parse_known_args()

    # ── merge YAML config ─────────────────────────────────────────────────────
    with open(args.c, "r") as f:
        yml = yaml.YAML(typ="rt")
        config = yml.load(f)
    for k, v in config.items():
        setattr(args, k, v)

    args.gpu_id = None
    return args


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── load class names ──────────────────────────────────────────────────────
    class_names = None
    try:
        common_config = args.common_config
        label_type = getattr(args, "label_type", "common")
        for path_key in ("train_data_config_path", "eval_data_config_path"):
            data_path = common_config.get(path_key)
            if data_path and os.path.exists(data_path):
                with open(data_path) as f:
                    data = json.load(f)
                names = []
                for key, value in data.items():
                    for v in value:
                        if v[label_type] not in names:
                            names.append(v[label_type])
                if names:
                    class_names = names
                    print(f"[info] Loaded {len(class_names)} class names from {data_path}")
                    break
    except Exception as e:
        print(f"[warn] Could not load class names: {e}")

    if not class_names:
        class_names = [f"class_{i}" for i in range(10)]
        print(f"[warn] Using {len(class_names)} dummy class names (data files not accessible).")
        print(f"       Head size will be {len(class_names)} instead of the real value; all other counts are exact.")

    # ── build classifier ──────────────────────────────────────────────────────
    print(f"[info] Building model (device={args.device}) ...")
    logging.basicConfig(level=logging.WARNING)  # suppress verbose build logs

    classifier = build_classifier(args, class_names, args.device)
    classifier.eval()
    print(f"[info] Model built successfully.\n")

    # ── print report ──────────────────────────────────────────────────────────
    print_report(classifier, args, detailed=getattr(args, "detailed", False))


if __name__ == "__main__":
    main()
