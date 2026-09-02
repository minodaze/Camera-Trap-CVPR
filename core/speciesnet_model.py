import json
import math
import os
import logging
from typing import Dict, List, Union, Any

import torch
from torch import nn
import torchvision

import timm

class LoRAConv2d(nn.Module):
    """
    Low-Rank Adaptation wrapper for 1x1 nn.Conv2d.
    Base conv weights stay trainable; only the low-rank bottleneck path is trained.
    """

    def __init__(self, conv: nn.Conv2d, r: int, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if conv.kernel_size != (1, 1):
            raise ValueError("LoRAConv2d currently supports only 1x1 convolutions.")
        if conv.groups != 1:
            raise ValueError("LoRAConv2d currently supports only non-grouped convolutions.")

        self.conv = conv
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        self.lora_down = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=r,
            kernel_size=1,
            stride=conv.stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class SpeciesNetClassifier(nn.Module):
    """
    Wrap SpeciesNet so it can be used by this repo's existing eval() function.

    Existing eval() expects:
        logits: [B, num_dataset_classes]
        labels: [B] with dataset class indices

    SpeciesNet raw model outputs:
        raw logits: [B, num_speciesnet_taxonomy_classes]

    This wrapper maps SpeciesNet taxonomy logits back to dataset class logits.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dataset_class_name_idx: Union[Dict[str, int], List[str]],
        speciesnet_aliases: Dict[str, List[str]],
        device: str,
        trainable: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        if isinstance(dataset_class_name_idx, list):
            dataset_class_name_idx = {c: i for i, c in enumerate(dataset_class_name_idx)}

        self.checkpoint_path = checkpoint_path
        self.dataset_class_name_idx = dataset_class_name_idx
        self.speciesnet_aliases = speciesnet_aliases
        self.device = device

        self.speciesnet, self.speciesnet_class_names = self._load_speciesnet(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.trainable = trainable
        self.lora_r = lora_r

        # name -> list of SpeciesNet logit indices.
        # This is safer than name -> idx because taxonomy may contain duplicate common names.
        self.speciesnet_name_to_idx = {}
        for i, name in enumerate(self.speciesnet_class_names):
            name_norm = self._norm_name(name)
            self.speciesnet_name_to_idx.setdefault(name_norm, []).append(i)

        self.dataset_to_speciesnet_indices = self._build_mapping()

        self._configure_training_mode(
            trainable=trainable,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        logging.info(f"[SpeciesNet] checkpoint: {checkpoint_path}")
        logging.info(f"[SpeciesNet] #speciesnet classes from taxonomy: {len(self.speciesnet_class_names)}")
        logging.info(f"[SpeciesNet] #unique speciesnet names: {len(self.speciesnet_name_to_idx)}")
        logging.info(f"[SpeciesNet] #dataset classes: {len(self.dataset_class_name_idx)}")
        logging.info(f"[SpeciesNet] dataset classes: {list(self.dataset_class_name_idx.keys())}")

    def _configure_training_mode(
        self,
        trainable: bool,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        if trainable:
            self.speciesnet.train()
            for p in self.speciesnet.parameters():
                p.requires_grad = True
            logging.info("[SpeciesNet] Full fine-tuning enabled.")
            self._log_trainable_summary()
            return

        for p in self.speciesnet.parameters():
            p.requires_grad = False

        if lora_r > 0:
            replaced = self._inject_lora_modules(
                self.speciesnet,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            if replaced == 0:
                raise ValueError(
                    "[SpeciesNet] LoRA was requested, but no supported modules were found to wrap. "
                    "This SpeciesNet checkpoint appears to be an FX-traced EfficientNet-style model "
                    "whose executed graph may not expose replaceable Python submodules. The current "
                    "LoRA implementation targets 1x1 nn.Conv2d modules. Use --full "
                    "for full fine-tuning if this checkpoint remains non-adaptable."
                )

            trainable_params = sum(p.numel() for p in self.speciesnet.parameters() if p.requires_grad)
            if trainable_params == 0:
                raise ValueError(
                    "[SpeciesNet] LoRA was requested, but SpeciesNet still has zero trainable "
                    "parameters after injection. The traced checkpoint is not exposing adaptable "
                    "modules to the current LoRA wrapper."
                )

            logging.info(
                f"[SpeciesNet] Enabled LoRA on {replaced} supported modules "
                f"({trainable_params} trainable parameters)."
            )
            self._log_lora_summary()
            self._log_trainable_summary()

        self.speciesnet.eval()

    def _log_lora_summary(self) -> None:
        wrapped_modules = [
            name for name, child in self.speciesnet.named_modules()
            if isinstance(child, LoRAConv2d)
        ]

        if not wrapped_modules:
            logging.info("[SpeciesNet] No LoRAConv2d modules were found after injection.")
            return

        logging.info(
            f"[SpeciesNet] LoRA-wrapped modules ({len(wrapped_modules)}): "
            + ", ".join(wrapped_modules)
        )

    def _log_trainable_summary(self) -> None:
        trainable_names = [
            name for name, param in self.speciesnet.named_parameters()
            if param.requires_grad
        ]

        if not trainable_names:
            logging.info("[SpeciesNet] No trainable parameters found.")
            return

        logging.info(
            f"[SpeciesNet] Trainable parameters ({len(trainable_names)}): "
            + ", ".join(trainable_names)
        )

    def _inject_lora_modules(
        self,
        module: nn.Module,
        r: int,
        alpha: float,
        dropout: float,
    ) -> int:
        replaced = 0
        for child_name, child in list(module.named_children()):
            if (
                isinstance(child, nn.Conv2d)
                and child.kernel_size == (1, 1)
                and child.groups == 1
            ):
                setattr(module, child_name, LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
                continue
            replaced += self._inject_lora_modules(child, r=r, alpha=alpha, dropout=dropout)
        return replaced

    def train(self, mode: bool = True):
        super().train(mode)
        if self.trainable:
            self.speciesnet.train(mode)
        else:
            self.speciesnet.eval()
        return self

    @staticmethod
    def _norm_name(name: str) -> str:
        return str(name).lower().strip()

    def _load_speciesnet(self, checkpoint_path: str, device: str):
        """
        Return:
            model: nn.Module, model(images) -> raw SpeciesNet logits [B, K]
            class_names: list[str], class_names[k] matches raw logit k
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SpeciesNet checkpoint not found: {checkpoint_path}")

        weight_dir = os.path.dirname(checkpoint_path)
        logging.info(f"[SpeciesNet] Loading checkpoint from {checkpoint_path}")

        # PyTorch >= 2.6 defaults weights_only=True, which fails for GraphModule checkpoints.
        # Use weights_only=False only for trusted checkpoints.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logging.info(f"[SpeciesNet] ckpt type: {type(ckpt)}")

        if isinstance(ckpt, dict):
            logging.info(f"[SpeciesNet] ckpt keys: {list(ckpt.keys())}")

        # Case 1: checkpoint itself is a full torch-saved model.
        if isinstance(ckpt, nn.Module):
            model = ckpt.to(device)
            class_names = self._extract_class_names_from_ckpt_or_dir({}, weight_dir)
            return model, class_names

        # Case 2: checkpoint dict contains a full model object.
        if isinstance(ckpt, dict):
            for key in ["model", "classifier", "speciesnet"]:
                if key in ckpt and isinstance(ckpt[key], nn.Module):
                    model = ckpt[key].to(device)
                    class_names = self._extract_class_names_from_ckpt_or_dir(ckpt, weight_dir)
                    return model, class_names

        # Case 3: checkpoint is or contains a state_dict.
        if isinstance(ckpt, dict):
            state_dict = self._extract_state_dict(ckpt)
            if state_dict is not None:
                class_names = self._extract_class_names_from_ckpt_or_dir(ckpt, weight_dir)
                model = self._build_speciesnet_timm_model(
                    state_dict=state_dict,
                    num_classes=len(class_names),
                    device=device,
                )
                return model, class_names

        raise RuntimeError(
            "Unsupported SpeciesNet checkpoint format. "
            "Print checkpoint type/keys and inspect the weight directory."
        )

    def _extract_state_dict(self, ckpt: Dict[str, Any]):
        """
        Try common checkpoint layouts.
        """
        for key in [
            "state_dict",
            "model_state_dict",
            "classifier_state_dict",
            "net",
            "model_weights",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return self._clean_state_dict_keys(ckpt[key])

        # Sometimes the entire ckpt is already a state_dict.
        tensor_values = [torch.is_tensor(v) for v in ckpt.values()]
        if len(tensor_values) > 0 and all(tensor_values):
            return self._clean_state_dict_keys(ckpt)

        return None

    def _clean_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]):
        """
        Remove common prefixes from DDP/lightning checkpoints.
        """
        cleaned = {}
        prefixes = [
            "module.",
            "model.",
            "classifier.",
            "speciesnet.",
            "net.",
        ]

        for k, v in state_dict.items():
            new_k = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if new_k.startswith(p):
                        new_k = new_k[len(p):]
                        changed = True
            cleaned[new_k] = v

        return cleaned

    def _build_speciesnet_timm_model(
        self,
        state_dict: Dict[str, torch.Tensor],
        num_classes: int,
        device: str,
    ):
        """
        Fallback for state_dict checkpoints.
        Your current checkpoint is a torch.fx GraphModule, so this likely won't be used.
        """

        candidate_archs = [
            "tf_efficientnetv2_m.in21k",
            "tf_efficientnetv2_m",
            "efficientnetv2_m",
        ]

        last_error = None

        for arch in candidate_archs:
            try:
                logging.info(f"[SpeciesNet] Trying timm architecture: {arch}")

                model = timm.create_model(
                    arch,
                    pretrained=False,
                    num_classes=num_classes,
                )

                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                logging.info(f"[SpeciesNet] Loaded state_dict with arch={arch}")
                logging.info(f"[SpeciesNet] missing keys count: {len(missing)}")
                logging.info(f"[SpeciesNet] unexpected keys count: {len(unexpected)}")

                total_keys = len(state_dict)
                if total_keys > 0 and len(unexpected) / total_keys > 0.5:
                    logging.warning(
                        f"[SpeciesNet] More than 50% unexpected keys for arch={arch}. "
                        f"This may be the wrong architecture."
                    )

                return model.to(device)

            except Exception as e:
                last_error = e
                logging.warning(f"[SpeciesNet] Failed arch={arch}: {e}")

        raise RuntimeError(
            "Could not instantiate SpeciesNet architecture from state_dict. "
            f"Last error: {last_error}"
        )

    def _extract_class_names_from_ckpt_or_dir(self, ckpt: Dict[str, Any], weight_dir: str):
        """
        Try to get class names from checkpoint first, then from taxonomy files in weight_dir.
        """
        # 1. Try checkpoint keys.
        for key in [
            "class_names",
            "classes",
            "labels",
            "taxonomy",
            "idx_to_class",
            "id_to_name",
            "label_names",
        ]:
            if isinstance(ckpt, dict) and key in ckpt:
                try:
                    return self._normalize_class_name_object(ckpt[key])
                except Exception as e:
                    logging.warning(f"[SpeciesNet] Could not parse ckpt['{key}']: {e}")

        candidates = [
            "always_crop_99710272_22x8_v12_epoch_00148.labels.txt",
        ]

        for filename in candidates:
            path = os.path.join(weight_dir, filename)
            if not os.path.exists(path):
                continue

            logging.info(f"[SpeciesNet] Loading class names from {path}")

            if filename.endswith(".txt"):
                class_names = self._load_class_names_from_taxonomy_txt(path)
                logging.info(f"[SpeciesNet] Loaded {len(class_names)} class names from {path}")
                logging.info(f"[SpeciesNet] First 10 class names: {class_names[:10]}")
                return class_names

            with open(path, "r") as f:
                data = json.load(f)

            class_names = self._normalize_class_name_object(data)
            logging.info(f"[SpeciesNet] Loaded {len(class_names)} class names from {path}")
            logging.info(f"[SpeciesNet] First 10 class names: {class_names[:10]}")
            return class_names

        raise FileNotFoundError(
            f"Could not find SpeciesNet class/taxonomy names in {weight_dir}. "
            f"Expected one of {candidates}, or class names stored inside checkpoint."
        )

    def _load_class_names_from_taxonomy_txt(self, path: str) -> List[str]:
        """
        Load SpeciesNet classifier labels.

        Important:
            The model output index must match the line index in the labels file.
            Therefore, NEVER skip a non-empty line just because the parsed common name
            is empty. Use a placeholder instead to preserve alignment.
        """
        class_names = []

        with open(path, "r") as f:
            for row_idx, line in enumerate(f):
                raw_line = line.rstrip("\n")

                # Blank lines and comment lines are NOT skipped here because
                # the model output index must equal the line index in this file.
                # Instead we use a placeholder to preserve row alignment.
                if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                    class_names.append(f"__empty_row_{row_idx}")
                    continue

                line = raw_line.strip()

                if ";" in line:
                    parts = [p.strip() for p in line.split(";")]

                    # Usually: uuid;class;order;family;genus;species;common_name
                    name = parts[-1].strip()

                    # Critical: preserve row alignment even if common_name is empty.
                    if not name:
                        # Prefer species/genus/family fallback if available.
                        fallback = None
                        for p in reversed(parts[:-1]):
                            if p.strip():
                                fallback = p.strip()
                                break

                        if fallback:
                            name = f"__missing_common_name_row_{row_idx}_{fallback}"
                        else:
                            name = f"__missing_common_name_row_{row_idx}"

                elif "\t" in line:
                    parts = [p.strip() for p in line.split("\t")]
                    name = parts[-1].strip() if parts[-1].strip() else f"__missing_label_row_{row_idx}"

                elif "," in line:
                    parts = [p.strip() for p in line.split(",")]
                    name = parts[-1].strip() if parts[-1].strip() else f"__missing_label_row_{row_idx}"

                else:
                    name = line

                class_names.append(name)

        if len(class_names) == 0:
            raise ValueError(f"No class names parsed from label file: {path}")

        logging.info(f"[SpeciesNet] Parsed {len(class_names)} labels from {path}")
        logging.info(f"[SpeciesNet] First 20 labels: {class_names[:20]}")

        return class_names

    def _normalize_class_name_object(self, obj):
        """
        Convert many possible taxonomy formats to list[str].
        """
        if isinstance(obj, list):
            if len(obj) == 0:
                raise ValueError("Empty class-name list")

            # ["plains zebra", ...]
            if isinstance(obj[0], str):
                return obj

            # [{"name": "plains zebra"}, ...]
            if isinstance(obj[0], dict):
                for field in [
                    "name",
                    "common_name",
                    "label",
                    "class_name",
                    "taxon",
                    "taxon_name",
                ]:
                    if field in obj[0]:
                        return [x[field] for x in obj]

                for field in ["common", "species", "scientific_name"]:
                    if field in obj[0]:
                        return [x[field] for x in obj]

                raise ValueError(f"Unknown list[dict] taxonomy fields: {obj[0].keys()}")

        if isinstance(obj, dict):
            # Nested dict: {"labels": [...]} etc.
            for key in [
                "class_names",
                "classes",
                "labels",
                "taxonomy",
                "categories",
            ]:
                if key in obj:
                    return self._normalize_class_name_object(obj[key])

            # idx -> name
            if all(str(k).isdigit() for k in obj.keys()):
                return [obj[str(i)] if str(i) in obj else obj[i] for i in range(len(obj))]

            # name -> idx
            if all(isinstance(v, int) for v in obj.values()):
                return [name for name, _ in sorted(obj.items(), key=lambda kv: kv[1])]

            # Sometimes dict values are taxonomy entries.
            values = list(obj.values())
            if len(values) > 0 and isinstance(values[0], dict):
                return self._normalize_class_name_object(values)

        raise ValueError(f"Unsupported class-name object format: {type(obj)}")

    def _build_mapping(self):
        mapping = {}

        for dataset_name, dataset_idx in self.dataset_class_name_idx.items():
            aliases = self.speciesnet_aliases.get(dataset_name, None)

            if aliases is None:
                raise ValueError(
                    f"Missing speciesnet_aliases entry for dataset class '{dataset_name}'.\n"
                    f"Dataset classes are: {list(self.dataset_class_name_idx.keys())}\n"
                    f"Alias keys are: {list(self.speciesnet_aliases.keys())}"
                )

            if isinstance(aliases, str):
                aliases = [a.strip() for a in aliases.split(",")]

            sn_indices = []

            for alias in aliases:
                alias_norm = self._norm_name(alias)

                if alias_norm == "skip":
                    continue

                if alias_norm not in self.speciesnet_name_to_idx:
                    candidates = [
                        name for name in self.speciesnet_name_to_idx.keys()
                        if alias_norm in name or name in alias_norm
                    ][:20]

                    raise ValueError(
                        f"SpeciesNet alias '{alias}' for dataset class '{dataset_name}' "
                        f"not found in SpeciesNet class names.\n"
                        f"Nearby candidates: {candidates}"
                    )

                # name -> list[index], because duplicate common names may exist.
                sn_indices.extend(self.speciesnet_name_to_idx[alias_norm])

            # Deduplicate while preserving order.
            sn_indices = list(dict.fromkeys(sn_indices))
            mapping[dataset_name] = sn_indices

            mapped_names = [self.speciesnet_class_names[i] for i in sn_indices]
            logging.info(f"[SpeciesNet mapping] {dataset_name} -> indices {sn_indices}, names {mapped_names}")

        return mapping

    def forward(self, images, return_feats: bool = False):
        # SpeciesNet exported GraphModule expects channels-last input: [B, H, W, C].
        # CkpDataset/ToTensor gives channels-first input: [B, C, H, W].
        if images.ndim == 4 and images.shape[1] in (3, 4):
            images = images.permute(0, 2, 3, 1).contiguous()

        # if self.training:
        #     # During training, EfficientNetV2-M retains ALL 400+ intermediate spatial
        #     # feature maps for backprop (e.g. [B,24,240,240], [B,48,120,120], ...),
        #     # consuming ~74 GB at batch=128. Gradient checkpointing recomputes them
        #     # during backward instead of storing them, at the cost of one extra forward.
        #     trainable_params = [p for p in self.speciesnet.parameters() if p.requires_grad]

        #     def _run_speciesnet(images, *_trainable_params):
        #         return self.speciesnet(images)

        #     raw_output = torch.utils.checkpoint.checkpoint(
        #         _run_speciesnet, images, *trainable_params, use_reentrant=False
        #     )
        # else:
        #     # During eval, torch.no_grad() is already active from the outer eval() context,
        #     # so a plain call is sufficient — no activations are stored.
        raw_output = self.speciesnet(images)

        # Some models return dict/tuple. Normalize to logits tensor.
        if isinstance(raw_output, dict):
            for key in ["logits", "classification", "classifier_logits", "predictions"]:
                if key in raw_output:
                    speciesnet_logits = raw_output[key]
                    break
            else:
                raise ValueError(f"Unknown SpeciesNet output keys: {raw_output.keys()}")

        elif isinstance(raw_output, (tuple, list)):
            speciesnet_logits = raw_output[0]

        else:
            speciesnet_logits = raw_output

        if speciesnet_logits.ndim != 2:
            raise ValueError(
                f"[SpeciesNet] Expected raw logits to have shape [B, K], "
                f"but got shape {tuple(speciesnet_logits.shape)}"
            )
        
        if not hasattr(self, "_printed_raw_topk"):
            x = speciesnet_logits.detach()

            logging.info(
                f"[SpeciesNet raw output] shape={tuple(x.shape)}, "
                f"min={x.min().item():.6f}, max={x.max().item():.6f}, "
                f"mean={x.mean().item():.6f}, sum_first={x[0].sum().item():.6f}"
            )

            # If output already sums to ~1 and values are all >=0, it is probably probabilities.
            if x.min().item() >= 0 and abs(x[0].sum().item() - 1.0) < 1e-3:
                scores = x
                logging.info("[SpeciesNet raw output] Looks like probabilities.")
            else:
                scores = torch.softmax(x, dim=1)
                logging.info("[SpeciesNet raw output] Looks like logits; applying softmax for debug only.")

            vals, inds = scores[:8].topk(10, dim=1)

            for b in range(min(8, x.shape[0])):
                logging.info(f"[SpeciesNet raw top10] sample {b}")
                for score, idx in zip(vals[b].tolist(), inds[b].tolist()):
                    logging.info(f"  {idx}: {self.speciesnet_class_names[idx]} | {score:.6f}")

            self._printed_raw_topk = True

        batch_size = speciesnet_logits.shape[0]
        num_speciesnet_logits = speciesnet_logits.shape[1]
        num_dataset_classes = len(self.dataset_class_name_idx)

        logging.debug(
            f"[SpeciesNet] raw logits shape: {tuple(speciesnet_logits.shape)}, "
            f"dataset classes: {num_dataset_classes}, taxonomy classes: {len(self.speciesnet_class_names)}"
        )

        # Validate indices and build dataset_idx -> sn_indices lookup.
        idx_to_sn: Dict[int, List[int]] = {}
        for dataset_name, dataset_idx in self.dataset_class_name_idx.items():
            sn_indices = self.dataset_to_speciesnet_indices[dataset_name]

            if len(sn_indices) == 0:
                idx_to_sn[dataset_idx] = []
                continue

            bad_indices = [i for i in sn_indices if i < 0 or i >= num_speciesnet_logits]
            if bad_indices:
                mapped_names = [
                    self.speciesnet_class_names[i]
                    for i in sn_indices
                    if 0 <= i < len(self.speciesnet_class_names)
                ]

                raise IndexError(
                    f"[SpeciesNet] Logit index out of bounds for dataset class '{dataset_name}'.\n"
                    f"speciesnet_logits.shape = {tuple(speciesnet_logits.shape)}\n"
                    f"num_speciesnet_logits = {num_speciesnet_logits}\n"
                    f"sn_indices = {sn_indices}\n"
                    f"bad_indices = {bad_indices}\n"
                    f"mapped_names = {mapped_names}\n"
                    f"taxonomy length = {len(self.speciesnet_class_names)}\n"
                    f"This means taxonomy_release.txt line order/length does not match the model output head."
                )

            idx_to_sn[dataset_idx] = sn_indices

        # Build each column and stack. Using torch.stack instead of in-place assignment
        # (dataset_logits[:, i] = ...) preserves gradient flow when training is enabled.
        # In-place writes into a new_full() tensor sever the autograd graph.
        cols = []
        for i in range(num_dataset_classes):
            sn_indices = idx_to_sn.get(i, [])
            if len(sn_indices) == 0:
                cols.append(speciesnet_logits.new_full((batch_size,), -100.0))
            else:
                cols.append(speciesnet_logits[:, sn_indices].max(dim=1).values)
        dataset_logits = torch.stack(cols, dim=1)

        if return_feats:
            return dataset_logits, None

        return dataset_logits

    def forward_features(self, images):
        raise NotImplementedError(
            "SpeciesNetClassifier.forward_features is not implemented. "
            "Use SpeciesNet for zero-shot eval only."
        )


# ---------------------------------------------------------------------------
# Dataset-level SpeciesNet utilities
# ---------------------------------------------------------------------------

def get_speciesnet_skip_classes(speciesnet_aliases):
    """
    Return the set of dataset class names whose SpeciesNet aliases are all "skip".

    In the YAML config, mark a class as excluded like:
        speciesnet_aliases:
          domestic animal: ["skip"]
          unknown: "skip"

    Returns an empty set if speciesnet_aliases is None or empty.
    """
    skip_classes = set()

    if not speciesnet_aliases:
        return skip_classes

    for cls_name, aliases in speciesnet_aliases.items():
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(",")]

        aliases_norm = [
            str(a).lower().strip()
            for a in aliases
            if str(a).strip()
        ]

        if len(aliases_norm) > 0 and all(a == "skip" for a in aliases_norm):
            skip_classes.add(cls_name)

    return skip_classes


def filter_dataset_by_class_names(dataset, class_names_to_skip, tag="dataset"):
    """
    Remove samples whose dataset class name is in *class_names_to_skip*.

    The class_name_idx mapping is kept intact; skipped labels simply no longer
    appear in the remaining samples.
    """
    if not class_names_to_skip:
        return dataset

    skip_indices = {
        dataset.class_name_idx[name]
        for name in class_names_to_skip
        if name in dataset.class_name_idx
    }

    missing = [
        name for name in class_names_to_skip
        if name not in dataset.class_name_idx
    ]

    if missing:
        logging.warning(
            f"[SpeciesNet] Skip classes not found in {tag} class_name_idx: {missing}"
        )

    old_len = len(dataset.samples)
    dataset.samples = [
        sample for sample in dataset.samples
        if sample.label not in skip_indices
    ]
    new_len = len(dataset.samples)

    logging.info(
        f"[SpeciesNet] Filtered skip classes from {tag}: "
        f"{sorted(class_names_to_skip)} | "
        f"indices={sorted(skip_indices)} | "
        f"{old_len} -> {new_len} samples"
    )

    return dataset