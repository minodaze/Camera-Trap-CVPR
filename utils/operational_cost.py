"""Operational cost measurement utilities for StreamTrap model updates."""

import csv
import json
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch


def count_parameters(model: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for param in model.parameters():
        ptr = param.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += param.numel()
    return total


def count_trainable_parameters(model: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            ptr = param.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += param.numel()
    return total


def get_trainable_parameter_names(model: torch.nn.Module) -> List[Tuple[str, int]]:
    seen = set()
    result = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            ptr = param.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                result.append((name, param.numel()))
    return result


def measure_checkpoint_size_mb(state_dict: dict) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(state_dict, tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        os.unlink(tmp_path)
    return size_mb


def get_gpu_memory_stats(device=None) -> Dict:
    if not torch.cuda.is_available():
        return {
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
            "gpu_name": "N/A",
            "cuda_version": "N/A",
        }
    if device is None:
        device = torch.cuda.current_device()
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(device)
    cuda_version = torch.version.cuda or "N/A"
    return {
        "peak_allocated_gb": round(peak_alloc, 4),
        "peak_reserved_gb": round(peak_reserved, 4),
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
    }


def write_results_csv(results: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


def write_results_json(results: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    existing = []
    if os.path.isfile(output_path):
        try:
            with open(output_path, "r") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, ValueError):
            existing = []
    existing.append(results)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)


def _build_petl_tag(args) -> str:
    tags = []
    if getattr(args, "lora_bottleneck", 0):
        tags.append(f"LoRA-r{args.lora_bottleneck}")
    if getattr(args, "ft_attn_module", None):
        tags.append(f"Adapter-attn{getattr(args, 'adapter_bottleneck', 64)}")
    if getattr(args, "ft_mlp_module", None):
        tags.append(f"Adapter-mlp{getattr(args, 'adapter_bottleneck', 64)}")
    if getattr(args, "vpt_mode", None):
        tags.append(f"VPT-{args.vpt_mode}")
    if getattr(args, "ssf", False):
        tags.append("SSF")
    if getattr(args, "fact_type", None):
        tags.append(f"FacT-{args.fact_type}")
    if getattr(args, "vqt_num", 0):
        tags.append(f"VQT-{args.vqt_num}")
    if getattr(args, "difffit", False):
        tags.append("DiffFit")
    if getattr(args, "full", False):
        tags.append("FullFT")
    return "+".join(tags) if tags else "Frozen"


class OperationalCostMeasurer:
    """Context manager that wraps a cl_module.process() call to measure operational cost."""

    def __init__(
        self,
        classifier,
        ckp_train_dset,
        al_mask,
        cl_config: dict,
        common_config: dict,
        args,
        ckp,
        output_dir: str,
        method_name: str,
    ):
        self.classifier = classifier
        self.ckp_train_dset = ckp_train_dset
        self.al_mask = al_mask
        self.cl_config = cl_config
        self.common_config = common_config
        self.args = args
        self.ckp = ckp
        self.output_dir = output_dir
        self.method_name = method_name
        self._start_time = None

    def __enter__(self):
        if torch.cuda.is_available():
            device = next(
                (p.device for p in self.classifier.parameters()), torch.device("cuda")
            )
            torch.cuda.reset_peak_memory_stats(device)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start_time

        if exc_type is not None:
            return False  # re-raise

        try:
            self._collect_and_save(elapsed)
        except Exception as e:
            import logging
            logging.warning(f"[OperationalCostMeasurer] Failed to collect stats: {e}")

        return False

    def _collect_and_save(self, elapsed_sec: float):
        import logging

        args = self.args
        cl_config = self.cl_config
        common_config = self.common_config
        classifier = self.classifier

        device = next(
            (p.device for p in classifier.parameters()), None
        )
        mem_stats = get_gpu_memory_stats(device)

        total_params = count_parameters(classifier)
        trainable_params = count_trainable_parameters(classifier)
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

        # Checkpoint size: adapter-only (trainable params only)
        trainable_names = {name for name, _ in get_trainable_parameter_names(classifier)}
        trainable_state = {
            k: v for k, v in classifier.state_dict().items() if k in trainable_names
        }
        adapter_only_mb = measure_checkpoint_size_mb(trainable_state)
        full_mb = measure_checkpoint_size_mb(classifier.state_dict())

        # Dataset stats
        import numpy as np
        n_total_samples = len(self.ckp_train_dset.samples)
        if self.al_mask is not None:
            n_selected = int(np.sum(self.al_mask))
        else:
            n_selected = n_total_samples

        label_to_cls = {v: k for k, v in self.ckp_train_dset.class_name_idx.items()}
        class_counts: Dict[str, int] = {}
        for mask_val, sample in zip(
            self.al_mask if self.al_mask is not None else [True] * n_total_samples,
            self.ckp_train_dset.samples,
        ):
            if mask_val:
                cls_name = label_to_cls.get(sample.label, str(sample.label))
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        n_classes = len(class_counts)

        petl_tag = _build_petl_tag(args)
        loss_type = cl_config.get("loss_type", getattr(args, "loss_type", "ce"))
        epochs = cl_config.get("epochs", "?")
        lr = cl_config.get("lr", common_config.get("lr", "?"))
        batch_size = common_config.get("train_batch_size", "?")

        results = {
            "method_name": self.method_name,
            "checkpoint_id": self.ckp,
            "petl_tag": petl_tag,
            "loss_type": loss_type,
            "training_time_sec": round(elapsed_sec, 3),
            "peak_gpu_memory_allocated_gb": mem_stats["peak_allocated_gb"],
            "peak_gpu_memory_reserved_gb": mem_stats["peak_reserved_gb"],
            "gpu_name": mem_stats["gpu_name"],
            "cuda_version": mem_stats["cuda_version"],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": round(trainable_ratio, 6),
            "lora_bottleneck": getattr(args, "lora_bottleneck", 0),
            "adapter_bottleneck": getattr(args, "adapter_bottleneck", 0),
            "epochs": epochs,
            "learning_rate": lr,
            "train_batch_size": batch_size,
            "n_total_training_samples": n_total_samples,
            "n_selected_training_samples": n_selected,
            "n_classes": n_classes,
            "adapter_only_checkpoint_mb": round(adapter_only_mb, 3),
            "full_checkpoint_mb": round(full_mb, 3),
        }

        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "operational_cost_results.csv")
        json_path = os.path.join(self.output_dir, "operational_cost_results.json")
        class_counts_path = os.path.join(
            self.output_dir, f"class_counts_ckp{self.ckp}.json"
        )

        write_results_csv(results, csv_path)
        write_results_json(results, json_path)
        with open(class_counts_path, "w") as f:
            json.dump({"checkpoint": self.ckp, "class_counts": class_counts}, f, indent=2)

        self._print_summary(results)
        logging.info(
            f"[OperationalCost] Results saved → {csv_path}, {json_path}, {class_counts_path}"
        )

    @staticmethod
    def _print_summary(r: dict):
        import logging
        sep = "─" * 62
        lines = [
            "",
            sep,
            f"  Operational Cost Report  |  {r['method_name']}  |  ckp {r['checkpoint_id']}",
            sep,
            f"  PETL method            : {r['petl_tag']}",
            f"  Loss type              : {r['loss_type']}",
            f"  Training time          : {r['training_time_sec']:.1f} s  ({r['training_time_sec']/60:.2f} min)",
            f"  Peak GPU memory (alloc): {r['peak_gpu_memory_allocated_gb']:.3f} GB",
            f"  Peak GPU memory (res.) : {r['peak_gpu_memory_reserved_gb']:.3f} GB",
            f"  GPU                    : {r['gpu_name']}",
            f"  Total parameters       : {r['total_parameters']:,}",
            f"  Trainable parameters   : {r['trainable_parameters']:,}  ({r['trainable_ratio']*100:.2f}%)",
            f"  LoRA bottleneck r      : {r['lora_bottleneck']}",
            f"  Adapter bottleneck     : {r['adapter_bottleneck']}",
            f"  Epochs                 : {r['epochs']}",
            f"  Learning rate          : {r['learning_rate']}",
            f"  Batch size             : {r['train_batch_size']}",
            f"  Training samples       : {r['n_selected_training_samples']} / {r['n_total_training_samples']} selected",
            f"  Classes                : {r['n_classes']}",
            f"  Adapter-only ckpt size : {r['adapter_only_checkpoint_mb']:.2f} MB",
            f"  Full model ckpt size   : {r['full_checkpoint_mb']:.2f} MB",
            sep,
            "",
        ]
        logging.info("\n".join(lines))
