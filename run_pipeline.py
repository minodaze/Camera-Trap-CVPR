import argparse
import code
import logging
import os
import copy
import time
import json
import re
import random

from datetime import datetime
from collections import defaultdict
import numpy as np
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader
import pprint
import pickle
import wandb  # Ensure wandb is imported
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from core import *
from core.module import get_al_module, get_cl_module, get_ood_module
from core.calibration_new.calibration_new import run_alg_calibration

from utils.misc import method_name
from utils.gpu_monitor import get_gpu_monitor, log_gpu_memory, monitor_model_memory
from utils.log_formatter import (
    setup_colored_logging, log_section_start, log_subsection_start, log_step,
    log_success, log_warning, log_error, log_info, log_config_item,
    log_checkpoint, log_final_result, log_metric, create_info_box, Colors,
    configure_colors_for_wandb
)
from plot.plot_features import plot_features
from plot.plot_text_F import plot_text_F

# Global worker init function for deterministic DataLoader behavior
def worker_init_fn(worker_id):
    # This will be set by the main process
    import os
    seed = int(os.environ.get('WORKER_SEED', 9527))
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

def setup_logging(log_path, debug, params):
    """Setup logging for the training process.
    
        Args:
            log_path (str): Path to save logs.
            debug (bool): Whether to run in debug mode.
        Returns:
            log_path (str): Path to save logs.
    
    """
    # Setup logging
    logger = logging.getLogger()
    
    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()
    
    petl_method_name = method_name(params)
    log_path = os.path.join(log_path, params.pretrained_weights)

    petl_method_name = petl_method_name + f'_text_{params.text}'
    if params.interpolation_model:
        petl_method_name += f'_interpolation_model_{params.interpolation_alpha}'
    if params.eval_only and params.merge_factor is not None:
        petl_method_name += f'_merge_factor_{params.merge_factor}'
    log_path = os.path.join(log_path, petl_method_name)
    if params.al_config and params.al_config.get('method') != 'none':
        al = params.al_config.get('method', 'none')
        if 'percentage' in params.al_config:
            al = al + f"_percentage_{params.al_config.get('percentage', 0)}"
        if 'num_sample_per_cluster' in params.al_config:
            al = al + f"_num_samples_per_cluster_{params.al_config.get('num_sample_per_cluster', 1)}"
        log_path = os.path.join(log_path, al)
    if not debug:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_path, 'log')
    else:
        logger.setLevel(logging.DEBUG)
        log_path = os.path.join(log_path, 'debug')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Log to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Log to file
    if log_path:
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, 'log.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Setup colored logging for console output
    setup_colored_logging()
    
    return log_path


def pretrain(classifier, class_names, pretrain_config, common_config, device, gpu_monitor=None, interpolation_model=False, interpolation_head=False, interpolation_alpha=0.5, eval_per_epoch=False, save_dir=None, args=None, test_per_epoch=False, eval_dset=None, ood_config=None, al_config=None):
    """
    Pretrain the classifier on the pretraining dataset.
    """
    # Get pretrain configurations
    pretrain_data_config_path = pretrain_config['pretrain_data_config_path']
    epochs = pretrain_config['epochs']
    optimizer_name = common_config['optimizer_name']
    optimizer_params = common_config['optimizer_params']
    train_batch_size = common_config['train_batch_size']
    
    _classifier = None  # Placeholder for interpolation model or head
    if pretrain_config['loss_type'] == 'kd' or interpolation_model or interpolation_head:
        _classifier = copy.deepcopy(classifier)
        _classifier.to(device)

    # Get optimizer
    optimizer = get_optimizer(classifier, optimizer_name, optimizer_params)
    logging.info(f'Pretraining with learning rate: {optimizer.param_groups[0]["lr"]:.8f}, optimizer: {optimizer_name}')
    
    # Get Scheduler
    if common_config['scheduler'] is not None:
        scheduler = get_scheduler(optimizer, common_config['scheduler'], common_config['scheduler_params'])
    else:
        scheduler = None

    # Get dataset
    dataset = CkpDataset(pretrain_data_config_path, class_names)
    dataset = dataset.get_subset(is_train=True, ckp_list=["ckp_-1", "ckp_1"])
    logging.info(f'Pretrain dataset size: {len(dataset)}. ')
    
    # Get evaluation dataset for pretrain eval_per_epoch if needed
    eval_loader = None
    if eval_per_epoch:
        # For pretraining, use 2 randomly selected images from each class across all checkpoint data
        # Load all training data (across all checkpoints)
        full_dataset = CkpDataset(pretrain_data_config_path, class_names)
        full_dataset = full_dataset.get_subset(is_train=True, ckp_list=["ckp_-1", "ckp_1"])
        
        from collections import defaultdict
        import random
        
        # Group samples by class
        class_to_samples = defaultdict(list)
        for i, sample in enumerate(full_dataset.samples):
            class_to_samples[sample.label].append(i)
        
        # Set random seed for reproducible splits
        random.seed(42)
        logging.info(f'Using fixed random seed (42) for validation split - this ensures consistent comparison across different LRs')
        
        train_indices = []
        val_indices = []
        
        # For pretraining: select exactly 2 images from each class for validation
        val_samples_per_class = 2
        logging.info(f'Using {val_samples_per_class} validation samples per class for pretraining (full training mode)')
        
        for class_label, sample_indices in class_to_samples.items():
            # Shuffle indices for this class
            shuffled_indices = sample_indices.copy()
            random.shuffle(shuffled_indices)
            
            # Select 2 samples for validation from each class
            if len(shuffled_indices) >= val_samples_per_class:
                class_val_indices = shuffled_indices[:val_samples_per_class]
                class_train_indices = shuffled_indices[val_samples_per_class:]
            else:
                # If a class has fewer than 2 samples, use 1 for validation
                class_val_indices = shuffled_indices[:1] if len(shuffled_indices) > 0 else []
                class_train_indices = shuffled_indices[1:] if len(shuffled_indices) > 1 else shuffled_indices
                logging.warning(f'Class {class_label} has only {len(shuffled_indices)} samples, using {len(class_val_indices)} for validation')
            
            val_indices.extend(class_val_indices)
            train_indices.extend(class_train_indices)
        
        # Shuffle the final indices to avoid class ordering
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        # Create validation dataset by filtering samples
        val_samples = [full_dataset.samples[i] for i in val_indices]
        eval_dataset = CkpDataset.__new__(CkpDataset)
        eval_dataset.is_crop = full_dataset.is_crop
        eval_dataset.is_train = False  # Use validation transforms
        eval_dataset.class_names = full_dataset.class_names
        eval_dataset.class_name_idx = full_dataset.class_name_idx
        eval_dataset.transform = full_dataset.val_transform
        eval_dataset.val_transform = full_dataset.val_transform
        eval_dataset.train_transform = full_dataset.train_transform
        eval_dataset.crop_train_transform = full_dataset.crop_train_transform
        eval_dataset.samples = val_samples
        eval_dataset.cache = full_dataset.cache
        
        # Update the main training dataset to exclude validation samples
        train_samples = [full_dataset.samples[i] for i in train_indices]
        dataset.samples = train_samples

        train_mask = np.ones(len(dataset), dtype=bool)  # Default to all samples
        # Do a random OOD selection if configured
        if ood_config and ood_config.get('method') != 'none':
            logging.info(f'Applying random OOD selection with fraction {ood_config.get("random_fraction", 0.5)}')
            ood_module = get_ood_module(ood_config, common_config, class_names, args, args.device)
            classifier, train_mask = ood_module.process(classifier, dataset, eval_dset, train_mask)
            dataset.apply_mask(train_mask)
        if al_config and al_config.get('method') != 'none':
            logging.info(f'Applying active learning selection with method {al_config.get("method")}')
            al_module = get_al_module(al_config, common_config, class_names, args, args.device)
            classifier, train_mask = al_module.process(classifier, dataset, eval_dset, train_mask, 'pretrain')
            dataset.apply_mask(train_mask)

        # Log class distribution in validation set with beautiful styling
        val_class_counts = defaultdict(int)
        train_class_counts = defaultdict(int)
        for sample in val_samples:
            val_class_counts[sample.label] += 1
        for sample in dataset.samples:
            train_class_counts[sample.label] += 1

        AL_summary = None
        if np.any(train_mask == 0):
            label_to_cls = {v: k for k, v in dataset.class_name_idx.items()}
            AL_summary = {
                'train_class_counts': {label_to_cls[k]: v for k, v in train_class_counts.items()},
                'total_val_samples': len(val_samples),
                'total_train_samples': len(dataset),
                'selected_stats': defaultdict(),
                'filtered_stats': defaultdict(),
                'selected_samples': defaultdict(list),
                'filtered_samples': defaultdict(list)
            }
            selected_cls = set()
            selected_cls_count = defaultdict(int)
            filtered_cls = set()
            filtered_cls_count = defaultdict(int)
            selected_n_samples_at_night = 0
            filtered_n_samples_at_night = 0
            for m, sample in zip(train_mask, dataset.samples):
                if m == 1:
                    # Convert datetime to string for JSON serialization
                    timestamp_str = sample.timestamp.isoformat() if sample.timestamp is not None else None
                    AL_summary['selected_samples'][label_to_cls[sample.label]].append((sample.file_path, timestamp_str))
                    if sample.timestamp is not None and (sample.timestamp.hour < 6 or sample.timestamp.hour > 20):
                        selected_n_samples_at_night += 1
                    selected_cls.add(label_to_cls[sample.label])
                    selected_cls_count[label_to_cls[sample.label]] += 1
                else:
                    # Convert datetime to string for JSON serialization
                    timestamp_str = sample.timestamp.isoformat() if sample.timestamp is not None else None
                    AL_summary['filtered_samples'][label_to_cls[sample.label]].append((sample.file_path, timestamp_str))
                    filtered_cls.add(label_to_cls[sample.label])
                    filtered_cls_count[label_to_cls[sample.label]] += 1
            total_selected = sum(len(samples) for samples in AL_summary['selected_samples'].values())
            total_filtered = sum(len(samples) for samples in AL_summary['filtered_samples'].values())
            AL_summary['selected_stats'] = {
                'num_selected': total_selected,
                'selected_cls': list(selected_cls),
                'selected_cls_count': selected_cls_count,
                'selected_n_samples_at_night': selected_n_samples_at_night
            }
            AL_summary['filtered_stats'] = {
                'num_filtered': total_filtered,
                'filtered_cls': list(filtered_cls),
                'filtered_cls_count': filtered_cls_count,
                'filtered_n_samples_at_night': filtered_n_samples_at_night
            }

        AL_summary_path = os.path.join(args.save_dir, 'active_learning_summary.json')
        if AL_summary is not None:
            with open(AL_summary_path, 'w') as f:
                json.dump(AL_summary, f, indent=2)
            log_success(f"Active Learning summary saved to {AL_summary_path}")

        # Create validation split summary with our theme
        log_subsection_start("📊 Validation Split Distribution", Colors.BRIGHT_CYAN)
        
        # Build detailed class distribution table
        class_distribution = ""
        for class_idx in sorted(set(list(val_class_counts.keys()) + list(train_class_counts.keys()))):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            train_count = train_class_counts[class_idx]
            val_count = val_class_counts[class_idx]
            total_count = train_count + val_count
            val_percentage = (val_count / total_count * 100) if total_count > 0 else 0
            class_distribution += f"{class_name:<20} {train_count:<8} {val_count:<8} {val_percentage:<8.1f}%\n"
        
        total_train = sum(train_class_counts.values())
        total_val = sum(val_class_counts.values())
        total_all = total_train + total_val
        overall_val_percentage = (total_val / total_all * 100) if total_all > 0 else 0
        
        # Create summary for info box
        validation_summary = f"Strategy: 2 samples per class for validation\n"
        validation_summary += f"Training samples: {total_train}\n"
        validation_summary += f"Validation samples: {total_val}\n"
        validation_summary += f"Validation percentage: {overall_val_percentage:.1f}%\n"
        validation_summary += f"Classes covered: {len(val_class_counts)}/{len(class_names)}"
        
        logging.info(create_info_box("Pretraining Validation Split", validation_summary))
        
        # Log detailed class distribution with colors
        log_info("📋 Per-class distribution:", Colors.CYAN)
        log_info(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Val%':<8}", Colors.BRIGHT_BLUE)
        log_info("─" * 50, Colors.BRIGHT_BLUE)
        
        for class_idx in sorted(set(list(val_class_counts.keys()) + list(train_class_counts.keys()))):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            train_count = train_class_counts[class_idx]
            val_count = val_class_counts[class_idx]
            total_count = train_count + val_count
            val_percentage = (val_count / total_count * 100) if total_count > 0 else 0
            
            # Use different colors for different validation percentages
            if val_percentage > 10:
                color = Colors.YELLOW  # High percentage (small class)
            elif val_percentage < 1:
                color = Colors.GREEN   # Low percentage (large class)
            else:
                color = Colors.WHITE   # Normal percentage
                
            log_info(f'{class_name:<20} {train_count:<8} {val_count:<8} {val_percentage:<8.1f}%', color)
        
        log_info("─" * 50, Colors.BRIGHT_BLUE)
        log_info(f'{"TOTAL":<20} {total_train:<8} {total_val:<8} {overall_val_percentage:<8.1f}%', Colors.BRIGHT_GREEN)
        
        eval_batch_size = common_config.get('eval_batch_size', train_batch_size)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
        
        log_success(f"Validation dataset prepared: {len(eval_dataset)} samples")
        log_success(f"Training dataset updated: {len(dataset)} samples")
    
    # Get loss function
    f_loss = get_f_loss(
        pretrain_config['loss_type'], 
        dataset.samples, 
        len(class_names),
        device,
        alpha=pretrain_config.get('loss_alpha', None),
        beta=pretrain_config.get('loss_beta', None),
        gamma=pretrain_config.get('loss_gamma', None),
        ref_model=_classifier,  # Use _classifier for KD loss if applicable
    )
    
    # Get dataloader

    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    # Get model saving setting from args, with fallback
    save_best_model = getattr(args, 'save_best_model', True) if args else True
    
    # Prepare test loaders for upper bound test_per_epoch (test on each checkpoint separately)
    next_test_loader = None
    if test_per_epoch and eval_dset is not None:
        # For upper bound: prepare individual test loaders for each checkpoint
        # Get checkpoint list directly from the evaluation dataset
        ckp_list = eval_dset.get_ckp_list()
        log_info(f"Available test checkpoints in dataset: {ckp_list}", Colors.CYAN)
        
        # Create individual test loaders for each checkpoint
        test_loaders = {}
        available_ckps = []
        for ckp in ckp_list:
            try:
                ckp_test_dset = eval_dset.get_subset(is_train=False, ckp_list=[ckp])
                if len(ckp_test_dset) > 0:
                    eval_batch_size = common_config.get('eval_batch_size', train_batch_size)
                    test_loader = DataLoader(ckp_test_dset, batch_size=eval_batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
                    test_loaders[ckp] = test_loader
                    available_ckps.append(ckp)
                    log_info(f"✅ Checkpoint {ckp}: {len(ckp_test_dset)} test samples", Colors.GREEN)
                else:
                    log_info(f"⚠️  Checkpoint {ckp}: no test samples found", Colors.YELLOW)
            except Exception as e:
                logging.warning(f"Could not load test data for checkpoint {ckp}: {e}")
        
        if test_loaders:
            # Store the dictionary of test loaders instead of a single loader
            next_test_loader = test_loaders
            total_samples = sum(len(loader.dataset) for loader in test_loaders.values())
            log_info(f"Upper bound test per epoch: individual test loaders for checkpoints {available_ckps} ({total_samples} total samples)", Colors.RED)
            log_info(f"TEST* computes average performance across {len(available_ckps)} individual test checkpoints", Colors.RED)
        else:
            log_info("Upper bound test per epoch: no test data available", Colors.YELLOW)
    # Train
    train(classifier, optimizer, loader, epochs, device, f_loss, 
          scheduler=scheduler, 
          gpu_monitor=gpu_monitor, 
          eval_per_epoch=eval_per_epoch, 
          eval_loader=eval_loader,
          save_best_model=save_best_model,
          save_dir=save_dir,
          model_name_prefix="pretrain",
          validation_mode=getattr(args, 'validation_mode', 'balanced_acc'),
          early_stop_epoch=getattr(args, 'early_stop_epoch', 5),
          test_per_epoch=test_per_epoch,
          next_test_loader=next_test_loader,
          test_type="UB_AVG" if test_per_epoch else "NEXT")

    if interpolation_model or interpolation_head:
        if gpu_monitor:
            gpu_monitor.log_memory_usage("interpolation", "after_pretrain")
        # Interpolate model if enabled
        if interpolation_model:
            logging.info(f'Interpolating model with alpha {interpolation_alpha}. ')
            classifier.interpolate_model(_classifier, alpha=interpolation_alpha)
        if interpolation_head:
            logging.info(f'Interpolating head with alpha {interpolation_alpha}. ')
            classifier.interpolate_head(_classifier, alpha=interpolation_alpha)
    del _classifier  # Clear the temporary classifier to free memory
    return classifier, AL_summary

def run(args):
    """Main execution workflow for the adaptive learning pipeline.
    
    Validation Strategies:
    1. For pretraining (full training): Uses 2 randomly selected images from each class 
       across all checkpoint data for validation
    2. For accumulative training: Uses current checkpoint's test data as validation 
       (e.g., when training on ckp_4, validation uses ckp_4's test data)
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        Returns:
            None
    
    """
    log_section_start("🚀 ICICLE BENCHMARK PIPELINE INITIALIZATION", Colors.BRIGHT_CYAN)
    
    # Initialize GPU memory monitoring if enabled
    gpu_monitor = None
    if args.gpu_memory_monitor:
        log_step(1, "Setting up GPU memory monitoring")
        enable_colors = not args.no_gpu_monitor_colors  # Colors enabled by default, disabled with --no_gpu_monitor_colors
        gpu_monitor = get_gpu_monitor(args.device, args.wandb, enable_colors)
        log_success("GPU memory monitoring enabled")
        gpu_monitor.log_memory_usage("startup", "initial")
    
    # Initialize wandb if enabled
    if args.wandb:
        log_step(2, "Initializing Weights & Biases logging")
        
        # Extract components from the original save_dir
        match = re.search(r"logs/([^/]+)/([^/]+)/([^/]+)/", getattr(args, "save_dir"))
        wandb_run_name = "Train"
        if match:
            dataset = match.group(1)
            setting = match.group(2)
            # Construct the wandb run name
            wandb_run_name = f"Train | {dataset} | {setting}"

        module_name = getattr(args, 'module_name', 'default_module')  # Fallback if module_name is not in args
        wandb.init(
            project="ICICLE Camera Trap CVPR",  # Replace with your project name
            name=wandb_run_name,  # Set run name using args.c and module_name
            config=vars(args)  # Log all arguments to wandb
        )
        log_success("Weights & Biases logging initialized")
        
        # Configure colors for wandb compatibility (darker colors for light background)
        configure_colors_for_wandb(wandb_enabled=True)
        log_info("Color scheme adjusted for wandb compatibility (darker colors)", Colors.CYAN)
    else:
        # Configure colors for terminal use (bright colors for dark background)
        configure_colors_for_wandb(wandb_enabled=False)

    log_subsection_start("📋 CONFIGURATION OVERVIEW")
    # Print args in a structured way
    config_summary = f"Device: {args.device}\n"
    config_summary += f"Seed: {args.seed}\n"
    config_summary += f"Validation Mode: {getattr(args, 'validation_mode', 'balanced_acc')}\n"
    config_summary += f"Early Stop Epochs: {getattr(args, 'early_stop_epoch', 5)}\n"
    logging.info(create_info_box("Pipeline Configuration", config_summary))
    
    # Display save directory separately for easy copying
    logging.info(f"{Colors.BRIGHT_BLUE}📁 Save Directory:{Colors.RESET} {args.save_dir}")
    
    # Log detailed configuration if debug
    if hasattr(args, 'debug') and args.debug:
        logging.info(f"\n{Colors.YELLOW}📋 Full Configuration:{Colors.RESET}")
        logging.info(pprint.pformat(vars(args)))
    
    log_step(3, "Loading data configuration")
    common_config = args.common_config
    pretrain_config = args.pretrain_config
    ood_config = args.ood_config
    al_config = args.al_config
    cl_config = args.cl_config
    train_path = common_config['train_data_config_path']
    test_path = common_config['eval_data_config_path']
    with open(train_path, 'r') as fin:
        data = json.load(fin)
    with open(test_path, 'r') as fin:
        data_test = json.load(fin)
        data.update(data_test)
    
    class_names = []
    label_type = args.label_type
    for key, value in data.items():
        for v in value:
            if v[label_type] not in class_names:
                class_names.append(v[label_type])
    del data, data_test  # Clear data to free memory

    rare_path = common_config.get('rare_data_config_path', None)
    if rare_path:
        with open(rare_path, 'r') as fin:
            data_rare = json.load(fin)
        for key, value in data_rare.items():
            for v in value:
                if v[label_type] not in class_names:
                    class_names.append(v[label_type])
        del data_rare  # Clear data to free memory
        log_info(f"Including rare classes from {rare_path}", Colors.CYAN)

    logging.info(f"Extracted {len(class_names)} unique classes from datasets using label type '{label_type}'")

    if args.expand_head:
        # Expand classes to include all classes from 0 to max class index
        with open(args.expand_head, 'r') as fin:
            expand_classes = fin.read().splitlines()  # Read lines from .txt file instead of json.load()
        class_names.extend([cls for cls in expand_classes if cls not in class_names])
        log_info(f"Expanded classes (total {len(class_names)} classes)", Colors.CYAN)

    is_crop = True if cl_config['method'] == 'co2l' else False
    
    log_success(f"Loaded {len(class_names)} classes using '{label_type}' labels")
    
    log_step(4, "Building classifier model")
    # Load model
    classifier = build_classifier(args, class_names, args.device)
    log_success(f"Classifier built successfully")
    # Monitor model memory usage if enabled
    if args.gpu_memory_monitor:
        gpu_monitor.log_memory_usage("model_load", "after_build")
        monitor_model_memory(classifier, "classifier", args.device, args.wandb)
        gpu_monitor.clear_cache_and_log("model_load")
    
    log_section_start("📊 DATASET PREPARATION", Colors.BRIGHT_YELLOW)
    
    # Prepare dataset
    train_dset = CkpDataset(common_config["train_data_config_path"], class_names, is_crop=is_crop, label_type=label_type)
    if rare_path:
        eval_dset = CkpDataset(rare_path, class_names, label_type=label_type)
    else:
        eval_dset = CkpDataset(common_config["eval_data_config_path"], class_names, label_type=label_type)
    
    # Monitor dataset memory usage if enabled
    if args.gpu_memory_monitor:
        gpu_monitor.log_memory_usage("dataset_load", "after_load", {
            'train_dataset_size': len(train_dset),
            'eval_dataset_size': len(eval_dset)
        })
    
    # Print ckp dict
    ckp_list = train_dset.get_ckp_list()
    
    dataset_summary = f"Total Training dataset size(all time span): {len(train_dset)}\n"
    dataset_summary += f"Total Evaluation dataset size(all time span): {len(eval_dset)}\n"
    dataset_summary += f"Number of checkpoints: {len(ckp_list)}\n"
    dataset_summary += f"Checkpoints: {', '.join(ckp_list)}"
    logging.info(create_info_box("Dataset Information", dataset_summary))
    
    log_section_start("🎯 PRETRAINING PHASE", Colors.BRIGHT_GREEN)
    
    AL_summary = None

    if args.plot_features and pretrain_config['pretrain']:
        try:
            result = plot_features(classifier, eval_dset, args, prefix='zs_before_pretrain_evalset_')
            if result[0] is None:
                log_warning("Feature plotting skipped for zero-shot eval dataset due to insufficient data")
        except Exception as e:
            log_warning(f"Feature plotting failed for zero-shot eval dataset: {str(e)}")
        
        try:
            result = plot_features(classifier, train_dset, args, prefix='zs_before_pretrain_trainset_')
            if result[0] is None:
                log_warning("Feature plotting skipped for zero-shot train dataset due to insufficient data")
        except Exception as e:
            log_warning(f"Feature plotting failed for zero-shot train dataset: {str(e)}")

    # Pretrain
    if pretrain_config['pretrain']:
        log_info(f"Pretraining enabled with {pretrain_config['epochs']} epochs")
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("pretrain", "before")
        classifier, AL_summary = pretrain(classifier, 
                              class_names, 
                              pretrain_config, 
                              common_config, 
                              args.device,
                              gpu_monitor,
                              interpolation_model=args.interpolation_model,
                              interpolation_head=args.interpolation_head,
                              interpolation_alpha=args.interpolation_alpha,
                              eval_per_epoch=args.eval_per_epoch,
                              save_dir=args.save_dir,
                              args=args,
                              test_per_epoch=args.test_per_epoch,
                              eval_dset=eval_dset,
                              al_config=al_config,
                              ood_config=ood_config)
        if args.plot_features:
            try:
                result = plot_features(classifier, eval_dset, args, prefix='after_pretrain_evalset_')
                if result[0] is None:
                    log_warning("Feature plotting skipped for post-pretrain eval dataset due to insufficient data")
            except Exception as e:
                log_warning(f"Feature plotting failed for post-pretrain eval dataset: {str(e)}")
            
            try:
                result = plot_features(classifier, train_dset, args, prefix='after_pretrain_trainset_')
                if result[0] is None:
                    log_warning("Feature plotting skipped for post-pretrain train dataset due to insufficient data")
            except Exception as e:
                log_warning(f"Feature plotting failed for post-pretrain train dataset: {str(e)}")

        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("pretrain", "after")
            gpu_monitor.clear_cache_and_log("pretrain")
        log_success("Pretraining completed successfully")
    else:
        log_warning("Pretraining skipped (disabled in configuration)")
    
    log_step(5, "Initializing pipeline modules")
    if args.plot_text_F:
        plot_text_F(classifier, class_names, args, prefix='text_feature_')
        # Note: Continue with normal execution after plotting
    
    # Initialize modules
    ood_module = get_ood_module(ood_config, common_config, class_names, args, args.device)
    al_module = get_al_module(al_config, common_config, class_names, args, args.device)
    cl_module = get_cl_module(classifier, cl_config, common_config, class_names, args, args.device)
    
    module_summary = f"OOD Method: {ood_config.get('method', 'none')}\n"
    module_summary += f"Active Learning Method: {al_config.get('method', 'none')}\n"
    module_summary += f"Continual Learning Method: {cl_config.get('method', 'none')}"
    logging.info(create_info_box("Module Configuration", module_summary))
    
    log_section_start("🔄 CONTINUAL LEARNING LOOP", Colors.BRIGHT_MAGENTA)
    
    # Initialize final results tracking
    final_eval_results = {}

    # FR mask
    FR_mask = None

    preds_image = {}
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    preds_image['stats'] = {
        "num_cls": len(class_names),
        "class_dist": {c: 0 for c in class_names},
    }
    AL_summary = {}
    if args.lora_interpolate:
        lora_inter_json = {ckp: {} for ckp in ckp_list}
    # Main loop
    for i in range(len(ckp_list)):
        # Get checkpoint
        ckp_prev = ckp_list[i - 1] if i > 0 else None
        ckp = ckp_list[i]
        preds_image[ckp] = {}
        
        log_subsection_start(f"📝 Processing Checkpoint {ckp} ({i+1}/{len(ckp_list)})", Colors.CYAN)
        
        # Monitor memory at checkpoint start
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("checkpoint", f"start_{ckp}")
        
        # Get training and evaluation dataset
        # For accumulative training: use all previous checkpoints (ckp_1 to ckp_(i-1))
        # prev_ckp_list = ckp_list[:i] if i > 0 else []
        ckp_train_dset = train_dset.get_subset(is_train=True, ckp_list=ckp_prev)
        ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp)
        logging.info(f'Training dataset size: {len(ckp_train_dset)}. ')
        logging.info(f'Evaluation dataset size: {len(ckp_eval_dset)}. ')
        
        # if ckp 
        # ckp_next_train_dset = train_dset.get_subset(is_train=True, ckp_list=ckp)
        # logging.info(f'Next training dataset size (for Feature Resonance): {len(ckp_next_train_dset)}.')

        train_cls_count = {}
        for sample in ckp_train_dset.samples:
            label = sample.label
            train_cls_count[label] = train_cls_count.get(label, 0) + 1
        sum_count = sum(train_cls_count.values())
        logging.info(f"Class distribution in training dataset on checkpoint {ckp_prev if ckp_prev is not None else 'NONE'}:")
        for cls_idx, count in train_cls_count.items():
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            logging.info(f'  Class {cls_name} (idx {cls_idx}): {count} samples {count / sum_count:.2%}')

        # Log evaluation dataset class distribution
        eval_cls_count = {}
        for sample in ckp_eval_dset.samples:
            label = sample.label
            eval_cls_count[label] = eval_cls_count.get(label, 0) + 1
        sum_count = sum(eval_cls_count.values())
        logging.info(f"Class distribution in evaluation dataset on checkpoint {ckp}:")
        for cls_idx, count in eval_cls_count.items():
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            logging.info(f'  Class {cls_name} (idx {cls_idx}): {count} samples {count / sum_count:.2%}')

        # Monitor memory after data subset
        if args.gpu_memory_monitor:
            gpu_monitor.monitor_data_loading(f"ckp_{ckp}_train", 
                                           common_config['train_batch_size'], 
                                           len(ckp_train_dset) // common_config['train_batch_size'])
            gpu_monitor.monitor_data_loading(f"ckp_{ckp}_eval", 
                                           common_config['eval_batch_size'], 
                                           len(ckp_eval_dset) // common_config['eval_batch_size'])
        
        # Initialize mask
        train_dset_mask = np.ones(len(ckp_train_dset), dtype=bool)
        
        if args.plot_features and not pretrain_config['pretrain']:
            try:
                result = plot_features(classifier, ckp_eval_dset, args, prefix=f'before_train_evalset{ckp}_')
                if result[0] is None:
                    log_warning(f"Feature plotting skipped for eval dataset {ckp} due to insufficient data")
            except Exception as e:
                log_warning(f"Feature plotting failed for eval dataset {ckp}: {str(e)}")
            
            if i != 0:  # Skip for first checkpoint to save time
                try:
                    result = plot_features(classifier, ckp_train_dset, args, prefix=f'before_train_trainset{ckp_prev}_')
                    if result[0] is None:
                        log_warning(f"Feature plotting skipped for train dataset {ckp_prev} due to insufficient data")
                except Exception as e:
                    log_warning(f"Feature plotting failed for train dataset {ckp_prev}: {str(e)}")

        # Run OOD detection
        # Check if first checkpoint
        is_first_ckp = (i == 0)
        is_zs = (args.module_name == 'zs')
        log_step(1, f"Out-of-Distribution Detection ({ood_config.get('method', 'none')})", Colors.YELLOW)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("ood", f"before_{ckp}")
        if not pretrain_config['pretrain']:
            classifier, ood_mask = ood_module.process(classifier, ckp_train_dset, ckp_eval_dset, train_dset_mask, is_first_ckp=is_first_ckp, is_zs=is_zs)
            log_info(f"OOD samples identified: {ood_mask.sum()} / {len(ood_mask)}", Colors.YELLOW)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("ood", f"after_{ckp}")   
        
        # Run active learning
        log_step(2, f"Active Learning ({al_config.get('method', 'none')})", Colors.BLUE)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("active_learning", f"before_{ckp}")
        if not pretrain_config['pretrain']:
            if al_config.get('method') == 'fr':
                # For FR: use the mask generated in the previous iteration for current training
                al_mask = copy.deepcopy(FR_mask) if FR_mask is not None else ood_mask
                
                # Generate FR_mask for the NEXT checkpoint
                next_train_dset = train_dset.get_subset(is_train=False, ckp_list=ckp)
                prev_ckps = ckp_list[:i] if i > 0 else []
                id_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=prev_ckps) if ckp_prev else None
                id_train_dset = copy.deepcopy(ckp_train_dset) if ckp_prev else None
                if ckp_prev:
                    id_train_dset.apply_mask(al_mask)
                    id_train_dset.add_samples(cl_module.buffer)
                    id_train_dset.eval()
                log_info(f"(FR) Active learning mask applied for current checkpoint: {al_mask.sum()} / {len(al_mask)}", Colors.BLUE)
                
                classifier, FR_mask = al_module.process(
                    classifier,
                    id_train_dset,
                    ckp_eval_dset,
                    ood_mask,
                    ckp=ckp,
                    id_eval_dset=id_eval_dset,
                    next_train_dset=next_train_dset
                )
                del id_train_dset, id_eval_dset, next_train_dset  # Clear to free memory
                log_info(f"(FR) new Active learning mask generated for next checkpoint: {FR_mask.sum()} / {len(FR_mask)}", Colors.BLUE)
            else:
                classifier, al_mask = al_module.process(
                    classifier, 
                    ckp_train_dset, 
                    ckp_eval_dset, 
                    ood_mask, 
                    ckp=ckp,
                )
                log_info(f"Active learning samples selected: {al_mask.sum()} / {len(al_mask)}", Colors.BLUE)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("active_learning", f"after_{ckp}")

        # Prepare evaluation dataloader
        cl_eval_loader = DataLoader(ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False, worker_init_fn=worker_init_fn)

        # Generate AL summary if active learning was applied
        if not pretrain_config['pretrain'] and 'al_mask' in locals():
            label_to_cls = {v: k for k, v in ckp_train_dset.class_name_idx.items()}
            train_class_counts = {label_to_cls[s.label]: 0 for s in ckp_train_dset.samples}
            for sample in ckp_train_dset.samples:
                train_class_counts[label_to_cls[sample.label]] += 1
            
            AL_summary[ckp] = {
                'train_class_counts': {k: v for k, v in train_class_counts.items()},
                'total_train_samples': len(ckp_train_dset.samples),
                'selected_stats': {},
                'filtered_stats': {},
                'selected_samples': defaultdict(list),
                'filtered_samples': defaultdict(list)
            }
            selected_cls = set()
            selected_cls_count = defaultdict(int)
            filtered_cls = set()
            filtered_cls_count = defaultdict(int)
            selected_n_samples_at_night = 0
            filtered_n_samples_at_night = 0
            for m, sample in zip(al_mask, ckp_train_dset.samples):
                if m == 1:
                    # Convert datetime to string for JSON serialization
                    timestamp_str = sample.timestamp.isoformat() if sample.timestamp is not None else None
                    AL_summary[ckp]['selected_samples'][label_to_cls[sample.label]].append((sample.file_path, timestamp_str))
                    if sample.timestamp is not None and (sample.timestamp.hour < 6 or sample.timestamp.hour > 20):
                        selected_n_samples_at_night += 1
                    selected_cls.add(label_to_cls[sample.label])
                    selected_cls_count[label_to_cls[sample.label]] += 1
                else:
                    # Convert datetime to string for JSON serialization
                    timestamp_str = sample.timestamp.isoformat() if sample.timestamp is not None else None
                    AL_summary[ckp]['filtered_samples'][label_to_cls[sample.label]].append((sample.file_path, timestamp_str))
                    filtered_cls.add(label_to_cls[sample.label])
                    filtered_cls_count[label_to_cls[sample.label]] += 1
            total_selected = sum(len(samples) for samples in AL_summary[ckp]['selected_samples'].values())
            total_filtered = sum(len(samples) for samples in AL_summary[ckp]['filtered_samples'].values())
            AL_summary[ckp]['selected_stats'] = {
                'num_selected': total_selected,
                'selected_cls': list(selected_cls),
                'selected_cls_count': selected_cls_count,
                'selected_n_samples_at_night': selected_n_samples_at_night
            }
            AL_summary[ckp]['filtered_stats'] = {
                'num_filtered': total_filtered,
                'filtered_cls': list(filtered_cls),
                'filtered_cls_count': filtered_cls_count,
                'filtered_n_samples_at_night': filtered_n_samples_at_night
            }

        # Prepare validation loader for continual learning
        cl_validation_loader = None

        if args.eval_per_epoch:
            # Check if we're using accumulative training (which trains incrementally)
            cl_method = cl_config.get('method', 'none')
            if 'accumulative' in cl_method and ckp_prev:
                # For accumulative training: use previous checkpoint's test data as validation
                # (we train on ckp_prev, so we validate on ckp_prev test data)
                ckp_validation_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp_prev)
                cl_validation_loader = DataLoader(ckp_validation_dset, batch_size=common_config['eval_batch_size'], shuffle=False, num_workers=4)
                log_info(f"Using previous checkpoint {ckp_prev} test data as validation for accumulative training", Colors.CYAN)
            else:
                # For other methods or first checkpoint: use current checkpoint's test data as validation
                cl_validation_loader = cl_eval_loader
                if 'accumulative' in cl_method:
                    log_info(f"Using current checkpoint {ckp} test data as validation (first checkpoint)", Colors.CYAN)
                else:
                    log_info(f"Using regular evaluation data as validation for {cl_method}", Colors.CYAN)
        
        # Run continual learning
        log_step(3, f"Continual Learning ({cl_config.get('method', 'none')})", Colors.GREEN)
        
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("continual_learning", f"before_{ckp}")
        if not pretrain_config['pretrain']:
            classifier = cl_module.process(
                classifier, 
                ckp_train_dset, 
                ckp_eval_dset, 
                al_mask, 
                eval_per_epoch=args.eval_per_epoch, 
                eval_loader=cl_validation_loader, 
                ckp=ckp,
                gpu_monitor=gpu_monitor if args.gpu_memory_monitor else None
            )
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("continual_learning", f"after_{ckp}")
        
        if args.plot_features and not pretrain_config['pretrain'] and i != 0:
            try:
                result = plot_features(classifier, ckp_eval_dset, args, prefix=f'after_trainon{ckp_prev}_evalset{ckp}_')
                if result[0] is None:
                    log_warning(f"Feature plotting skipped for post-training eval dataset {ckp} due to insufficient data")
            except Exception as e:
                log_warning(f"Feature plotting failed for post-training eval dataset {ckp}: {str(e)}")
            
            if i != 0:
                try:
                    result = plot_features(classifier, ckp_train_dset, args, prefix=f'after_trainon{ckp_prev}_trainset{ckp_prev}_')
                    if result[0] is None:
                        log_warning(f"Feature plotting skipped for post-training train dataset {ckp_prev} due to insufficient data")
                except Exception as e:
                    log_warning(f"Feature plotting failed for post-training train dataset {ckp_prev}: {str(e)}")

        # Force memory cleanup after continual learning
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if args.interpolation_model or args.interpolation_head:
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("interpolation", f"before_{ckp}")
            # Interpolate model if enabled
            if args.interpolation_model:
                log_info(f"Interpolating model at checkpoint {ckp} with alpha {args.interpolation_alpha}", Colors.MAGENTA)
                classifier.interpolate_model(cl_module._classifier, alpha=args.interpolation_alpha)
            if args.interpolation_head:
                log_info(f"Interpolating head at checkpoint {ckp} with alpha {args.interpolation_alpha}", Colors.MAGENTA)
                classifier.interpolate_head(cl_module._classifier, alpha=args.interpolation_alpha)

        log_checkpoint(f"Training completed for checkpoint {ckp_prev}")
        
        # Save model and predictions
        acc, balanced_acc = 0.0, 0.0  # Initialize metrics
        eval_loss = 0.0  # Initialize eval_loss
        
        log_step(4, "Model Evaluation", Colors.BRIGHT_GREEN)
        
        if args.accu_eval:
            log_info(f"Running accumulative evaluation starting from checkpoint {ckp_list[i]}")
            start = i
            if i != 0:
                start = i - 1
            for c in range(start, len(ckp_list)):
                eval_ckp = ckp_list[c]
                ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=eval_ckp)
                cl_eval_loader = DataLoader(ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False, worker_init_fn=worker_init_fn)
                if args.gpu_memory_monitor:
                    gpu_monitor.log_memory_usage("evaluation", f"before_{eval_ckp}")
                loss_arr, preds_arr, labels_arr, _, _ = eval(classifier, cl_eval_loader, args.device, chop_head=common_config['chop_head'])
                if args.gpu_memory_monitor:
                    gpu_monitor.log_memory_usage("evaluation", f"after_{eval_ckp}")
                a, b = print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"📊 Accu-eval {ckp_list[i]} → {eval_ckp}: ")
                
                # Log all accu-eval results to wandb, not just the first one
                if wandb.run is not None:
                    eval_loss_curr = np.mean(loss_arr)
                    logging.info(f"Logging accu-eval metrics to wandb: ckp={ckp_list[i]}, eval_ckp={eval_ckp}, acc={a:.4f}, balanced_acc={b:.4f}, loss={eval_loss_curr:.4f}")
                    wandb.log({
                        f"accu_eval/accuracy_{eval_ckp}": a,
                        f"accu_eval/balanced_accuracy_{eval_ckp}": b,
                        f"accu_eval/loss_{eval_ckp}": eval_loss_curr,
                        f"accu_eval/step": c,
                        f"accu_eval/training_checkpoint": ckp_list[i]
                    }, step=i * len(ckp_list) + c)  # Use unique step for each accu-eval
                
                if c == i:
                    acc, balanced_acc = a, b  # Only use first checkpoint metrics for main logging
                    eval_loss = eval_loss_curr
        else:
            if ckp_prev is not None:
                log_info(f"Evaluating on current training checkpoint {ckp_prev}")
                current_ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp_prev)
                current_cl_eval_loader = DataLoader(current_ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False)
                loss_arr, preds_arr, labels_arr, _, _ = eval(classifier, current_cl_eval_loader, args.device, chop_head=common_config['chop_head'])
                print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"📊 Current ckp {ckp_prev}: ")
            
            log_info(f"Evaluating on target checkpoint {ckp}")
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"before_{ckp}")
            loss_arr, preds_arr, labels_arr, pred_true, pred_false = eval(classifier, cl_eval_loader, args.device, chop_head=common_config['chop_head'])
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"after_{ckp}")
            acc, balanced_acc = print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"📊 Target ckp {ckp}: ")
            eval_loss = np.mean(loss_arr)
            if args.lora_interpolate and lora_inter_json is not None:
                log_info(f"Evaluating LoRA-interpolated model on target checkpoint {ckp} with different alpha values", Colors.MAGENTA)
                alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                for alpha in alphas:
                    classifier.interpolate_lora(alpha)
                    loss_arr, preds_arr, labels_arr, _, _ = eval(classifier, cl_eval_loader, args.device, chop_head=common_config['chop_head'])
                    a, b = print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"   🔄 LoRA alpha {alpha:.1f}: ")
                    lora_inter_json[ckp][f'alpha_{alpha:.1f}'] = {
                        'accuracy': float(a),
                        'balanced_accuracy': float(b),
                        'loss': float(np.mean(loss_arr))
                    }
            else:
                log_info(f"LoRA interpolation disabled or not configured; skipping LoRA-interpolated evaluations", Colors.MAGENTA)

        # Log metrics with colors
        log_metric("Accuracy", acc, ".4f", Colors.BRIGHT_GREEN)
        log_metric("Balanced Accuracy", balanced_acc, ".4f", Colors.BRIGHT_GREEN)
        log_metric("Evaluation Loss", eval_loss, ".4f", Colors.BRIGHT_BLUE)

        # Log training and evaluation loss to wandb
        if args.wandb:
            logging.info(f"📈 Logging metrics to W&B: ckp={ckp}, acc={acc:.4f}, balanced_acc={balanced_acc:.4f}, eval_loss={eval_loss:.4f}")
            wandb.log({
                "test/accuracy": acc,  # Overall accuracy
                "test/balanced_accuracy": balanced_acc,  # Balanced accuracy
                "test/loss": eval_loss,  # Evaluation loss
                "checkpoint": ckp
            }, step=i)

        # Update final evaluation results
        for sample in ckp_eval_dset.samples:
            label = sample.label
            preds_image[ckp].setdefault('class_count', {})
            preds_image[ckp]['class_count'][idx_to_class[label]] = preds_image[ckp]['class_count'].get(idx_to_class[label], 0) + 1
        final_eval_results[ckp] = {
            'num_samples': len(preds_arr),
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'loss': float(eval_loss),
            'num_samples': len(preds_arr)
        }

        preds_image[ckp]['balanced_accuracy'] = balanced_acc
        preds_image[ckp]['accuracy'] = acc
        
        cls_split_true = {}
        cls_split_false = {}
        ckp_true_confidence = 0.0
        ckp_false_confidence = 0.0
        cls_dist = {c: 0 for c in class_names}

        for idx, (file_path, label, pred, confidence) in enumerate(pred_true):
            ckp_true_confidence += confidence
            cls_split_true.setdefault("True label: " + idx_to_class[label], []).append({"file_path": file_path, "confidence": confidence})
            cls_dist[idx_to_class[label]] += 1
        for idx, (file_path, label, pred, confidence) in enumerate(pred_false):
            ckp_false_confidence += confidence
            cls_split_false.setdefault("True label: " + idx_to_class[label], []).append({"file_path": file_path, "prediction": idx_to_class[pred], "confidence": confidence})
            cls_dist[idx_to_class[label]] += 1

        preds_image[ckp]['true_confidence'] = ckp_true_confidence / max(1, len(pred_true))
        preds_image[ckp]['false_confidence'] = ckp_false_confidence / max(1, len(pred_false))
        preds_image[ckp]['avg_confidence'] = (ckp_true_confidence + ckp_false_confidence) / max(1, len(pred_true) + len(pred_false))
        preds_image[ckp]['class_count'] = cls_dist
        preds_image[ckp]['correct'] = cls_split_true
        preds_image[ckp]['incorrect'] = cls_split_false
        for c, count in cls_dist.items():
            preds_image['stats']['class_dist'][c] += count

        log_step(5, "Saving Results", Colors.MAGENTA)
        if args.is_save:
            save_path = os.path.join(args.save_dir, f'{ckp}.pth')
            torch.save(classifier.state_dict(), save_path)
            log_success(f"Model saved to {save_path}")
        
        pred_path = os.path.join(args.save_dir, f'{ckp}_preds.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump((preds_arr, labels_arr), f)
        log_success(f"Predictions saved to {pred_path}")
        
        if not pretrain_config['pretrain']:
            mask_path = os.path.join(args.save_dir, f'{ckp}_mask.pkl')
            with open(mask_path, 'wb') as f:
                pickle.dump((ood_mask, al_mask), f)
            log_success(f"Masks saved to {mask_path}")
        
        # Clear GPU cache between checkpoints
        if args.gpu_memory_monitor:
            gpu_monitor.clear_cache_and_log(f"checkpoint_end_{ckp}")
        
        log_subsection_start(f"✅ Checkpoint {ckp} Complete", Colors.BRIGHT_GREEN)

    # Save AL_summary after main loop completion
    AL_summary_path = os.path.join(args.save_dir, 'active_learning_summary.json')
    if AL_summary:
        with open(AL_summary_path, 'w') as f:
            json.dump(AL_summary, f, indent=2)
        log_success(f"Active Learning summary saved to {AL_summary_path}")
    else:
        logging.info("No active learning summary to save (AL not used or accumulative-scratch mode)")

    if args.lora_interpolate and lora_inter_json is not None:
        lora_inter_path = os.path.join(args.save_dir, 'lora_interpolation_results.json')
        with open(lora_inter_path, 'w') as f:
            json.dump(lora_inter_json, f, indent=2)
        log_success(f"LoRA interpolation results saved to {lora_inter_path}")

    # Calculate and report final average balanced accuracy across all checkpoints
    if final_eval_results:
        # Calculate average metrics
        avg_accuracy = sum(0 if np.isnan(result['accuracy']) else result['accuracy'] for result in final_eval_results.values()) / len(final_eval_results)
        avg_balanced_accuracy = sum(0 if np.isnan(result['balanced_accuracy']) else result['balanced_accuracy'] for result in final_eval_results.values()) / len(final_eval_results)
        avg_loss = sum(0 if np.isnan(result['loss']) else result['loss'] for result in final_eval_results.values()) / len(final_eval_results)
        total_samples = sum(result['num_samples'] for result in final_eval_results.values())
        
        # Log final summary
        print(Colors.BOLD + "=" * 80 + Colors.RESET)
        log_section_start("📊 FINAL EVALUATION SUMMARY", Colors.BRIGHT_YELLOW)
        logging.info("=== CHECKPOINT RESULTS SUMMARY ===")
        for ckp, results in final_eval_results.items():
            logging.info(f"{ckp}: acc={results['accuracy']:.4f}, balanced_acc={results['balanced_accuracy']:.4f}, "
                        f"loss={results['loss']:.4f}, samples={results['num_samples']}")
        
        logging.info("=" * 60)
        logging.info(f"🎯 FINAL AVERAGE ACROSS ALL CHECKPOINTS:")
        log_metric("Average Accuracy", avg_accuracy, ".4f", Colors.BRIGHT_GREEN)
        log_metric("Average Balanced Accuracy", avg_balanced_accuracy, ".4f", Colors.BRIGHT_GREEN)
        log_metric("Average Loss", avg_loss, ".4f", Colors.BRIGHT_BLUE)
        logging.info(f"  📊 Total Checkpoints: {len(final_eval_results)}")
        logging.info(f"  📊 Total Samples: {total_samples}")
        logging.info("=" * 60)
        
        # Log final average to wandb
        if args.wandb:
            wandb.log({
                "final/average_accuracy": avg_accuracy,
                "final/average_balanced_accuracy": avg_balanced_accuracy,
                "final/average_loss": avg_loss,
                "final/total_checkpoints": len(final_eval_results),
                "final/total_samples": total_samples
            })
        
        # Save final summary to file
        final_summary = {
            'checkpoint_results': final_eval_results,
            'averages': {
                'accuracy': float(avg_accuracy),
                'balanced_accuracy': float(avg_balanced_accuracy),
                'loss': float(avg_loss)
            },
            'total_checkpoints': len(final_eval_results),
            'total_samples': int(total_samples)
        }
        average_confidence = sum(preds_image[ckp]['avg_confidence'] for ckp in final_eval_results) / len(final_eval_results)
        preds_image['stats']['total_samples'] = total_samples
        preds_image['stats']['total_checkpoints'] = len(final_eval_results)
        preds_image['stats']['average'] = {
                'accuracy': float(avg_accuracy),
                'balanced_accuracy': float(avg_balanced_accuracy),
                'average_confidence': float(average_confidence),
                'loss': float(avg_loss)
            }
        
        summary_path = os.path.join(args.save_dir, 'final_training_summary.json')
        preds_image_path = os.path.join(args.save_dir, 'final_image_level_predictions.json')
        
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2)
        with open(preds_image_path, 'w') as f:
            json.dump(preds_image, f, indent=2)
        log_success(f"Final summary saved to {summary_path}")

    # Final completion logs
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    log_final_result("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    
    # Final summary with colors
    log_info(f"📊 Total checkpoints processed: {len(ckp_list)}")
    log_info(f"💾 Results saved to: {args.save_dir}")

    # Final GPU memory summary
    if args.gpu_memory_monitor:
        summary = gpu_monitor.get_memory_summary()
        log_info(f"🔧 GPU Memory Summary: {summary}")
        if args.wandb:
            wandb.log({"gpu_memory/summary": summary})

    # Finalize wandb if enabled
    if args.wandb:
        log_info(f"📈 Metrics logged to W&B project: {wandb.run.project if wandb.run else 'Unknown'}")
        wandb.finish()
        log_success("✅ W&B run finished")
    
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    print()

def run_eval_only(args):
    """Run evaluation only on trained model checkpoints.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    log_section_start("🔍 EVALUATION-ONLY MODE INITIALIZATION", Colors.BRIGHT_CYAN)
    
    # Initialize GPU memory monitoring if enabled
    gpu_monitor = None
    if args.gpu_memory_monitor:
        log_step(1, "Setting up GPU memory monitoring")
        enable_colors = not args.no_gpu_monitor_colors
        gpu_monitor = get_gpu_monitor(args.device, args.wandb, enable_colors)
        log_success("GPU memory monitoring enabled")
        gpu_monitor.log_memory_usage("startup", "initial")
    
    # Initialize wandb if enabled
    if args.wandb:
        log_step(2, "Initializing Weights & Biases logging")
        import re
        match = re.search(r"pipeline/([^/]+)/([^/]+)/([^/]+)", args.save_dir)
        wandb_run_name = "Eval Only Run"
        if match:
            dataset = match.group(1)
            setting = match.group(2)
            wandb_run_name = f"EVAL | {dataset} | {setting}"

        wandb.init(
            project="Camera Trap Benchmark - EVAL ONLY",
            name=wandb_run_name,
            config=vars(args)
        )
        log_success("Weights & Biases logging initialized")
        
        # Configure colors for wandb compatibility (darker colors for light background)
        configure_colors_for_wandb(wandb_enabled=True)
        log_info("Color scheme adjusted for wandb compatibility (darker colors)", Colors.CYAN)
    else:
        # Configure colors for terminal use (bright colors for dark background)
        configure_colors_for_wandb(wandb_enabled=False)
    
    log_subsection_start("📋 VALIDATION & SETUP")
    # Validate required arguments for eval_only mode
    if not args.model_dir:
        log_error("--model_dir is required for eval_only mode")
        raise ValueError("--model_dir is required for eval_only mode")
    
    if not os.path.exists(args.model_dir):
        log_error(f"Model directory not found: {args.model_dir}")
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    log_success(f"Model directory found: {args.model_dir}")

    # Optional: load weight sanity reference results
    weight_sanity_enabled = getattr(args, 'weight_sanity', False)
    sanity_expected = {}
    sanity_offset = 0.03
    sanity_mismatches = []
    sanity_total = 0
    if weight_sanity_enabled:
        ref_path = os.path.join(args.model_dir, 'final_training_summary.json')
        if not os.path.exists(ref_path):
            log_warning(f"Weight sanity requested, but reference file not found: {ref_path}")
            weight_sanity_enabled = False
        else:
            try:
                with open(ref_path, 'r') as rf:
                    ref_data = json.load(rf)
                # Build expected balanced_accuracy map per checkpoint
                ckps = ref_data.get('checkpoint_results', {})
                for k, v in ckps.items():
                    if isinstance(v, dict) and 'balanced_accuracy' in v:
                        sanity_expected[k] = float(v['balanced_accuracy'])
                if sanity_expected:
                    log_info(f"Weight sanity enabled. Loaded {len(sanity_expected)} checkpoint references from {ref_path}", Colors.CYAN)
                else:
                    log_warning("Weight sanity enabled but no checkpoint results found in reference file; disabling")
                    weight_sanity_enabled = False
            except Exception as e:
                log_warning(f"Failed to load weight sanity reference: {e}; disabling sanity check")
                weight_sanity_enabled = False
    
    log_step(3, "Loading data configuration")
    
    common_config = args.common_config
    train_path = common_config['train_data_config_path']
    test_path = common_config['eval_data_config_path']
    with open(train_path, 'r') as fin:
        data = json.load(fin)
    with open(test_path, 'r') as fin:
        data_test = json.load(fin)
        data.update(data_test)
    
    class_names = []
    label_type = args.label_type
    for key, value in data.items():
        for v in value:
            if v[label_type] not in class_names:
                class_names.append(v[label_type])
    del data, data_test  # Clear data to free memory

    rare_path = common_config.get('rare_data_config_path', None)
    if rare_path:
        with open(rare_path, 'r') as fin:
            data_rare = json.load(fin)
        for key, value in data_rare.items():
            for v in value:
                if v[label_type] not in class_names:
                    class_names.append(v[label_type])
        del data_rare  # Clear data to free memory
        log_info(f"Including rare classes from {rare_path}", Colors.CYAN)

    if args.expand_head:
        with open(args.expand_head, 'r') as fin:
            expand_classes = fin.read().splitlines()
        class_names.extend([c for c in expand_classes if c not in class_names])
        log_info(f"Expanded head with new classes: {expand_classes}", Colors.YELLOW)

    log_success(f"Loaded {len(class_names)} classes using '{label_type}' labels")
    
    log_step(4, "Building classifier model")
    # Build classifier architecture (same as training)
    classifier = build_classifier(args, class_names, args.device)
    log_success("Classifier built successfully")
    
    if args.gpu_memory_monitor:
        gpu_monitor.log_memory_usage("model_load", "after_build")
        monitor_model_memory(classifier, "classifier", args.device, args.wandb)
    
    log_section_start("📊 DATASET PREPARATION", Colors.BRIGHT_YELLOW)
    # Prepare dataset
    if rare_path:
        eval_dset = CkpDataset(rare_path, class_names, is_train=False, label_type=label_type)
    else:
        eval_dset = CkpDataset(common_config["eval_data_config_path"], class_names, is_train=False, label_type=label_type)
    
    # Prepare train dset for calibration
    train_dset = None
    checkpoint_analysis = None
    if args.calibration:
        
        train_dset = CkpDataset(common_config["train_data_config_path"], class_names, is_train=True, label_type=label_type)
        
        # Analyze checkpoint classes for seen/unseen
        with open(common_config["train_data_config_path"], 'r') as f:
            data = json.load(f)
        
        # Get all unique classes
        all_classes = set()
        for checkpoint, entries in data.items():
            for entry in entries:
                all_classes.add(entry[label_type])
        all_classes = sorted(list(all_classes))
        
        # Sort checkpoints by number
        checkpoints = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
        
        # Build cumulative seen classes
        cumulative_seen = set()
        checkpoint_analysis = {}
        
        for checkpoint in checkpoints:
            checkpoint_classes = set()
            for entry in data[checkpoint]:
                checkpoint_classes.add(entry[label_type])
            cumulative_seen.update(checkpoint_classes)
            seen = sorted(list(cumulative_seen))
            unseen = sorted(list(set(all_classes) - cumulative_seen))
            checkpoint_analysis[checkpoint] = {'seen': seen, 'unseen': unseen}\
        

    # Get checkpoint list
    ckp_list = eval_dset.get_ckp_list()
    log_info(f"Available checkpoints in dataset: {ckp_list}", Colors.CYAN)
    
    # Determine which checkpoints to evaluate and detect training mode
    if args.checkpoint_list:
        log_info(f"User specified checkpoints: {args.checkpoint_list}", Colors.CYAN)
        # Use specific checkpoints provided by user
        target_checkpoints = []
        for ckp_str in args.checkpoint_list:
            ckp_name = f"ckp_{ckp_str}"
            if ckp_name in ckp_list:
                target_checkpoints.append(ckp_name)
            else:
                log_warning(f"Checkpoint {ckp_name} not found in dataset. Available: {ckp_list}")
        
        if not target_checkpoints:
            log_error("No valid checkpoints found from the provided list")
            raise ValueError("No valid checkpoints found from the provided list")
        
        ckp_list = target_checkpoints
        log_success(f"Will evaluate specified checkpoints: {ckp_list}")
    else:
        ckp_list = eval_dset.get_ckp_list()
        log_info(f"Will evaluate all available checkpoints: {ckp_list}", Colors.CYAN)
    
    dataset_summary = f"Evaluation dataset size: {len(eval_dset)}\n"
    dataset_summary += f"Number of checkpoints: {len(ckp_list)}\n"
    dataset_summary += f"Checkpoints: {', '.join(ckp_list)}"
    logging.info(create_info_box("Dataset Information", dataset_summary))
    log_section_start("🔍 TRAINING MODE DETECTION", Colors.BRIGHT_GREEN)
    # Detect training mode by checking what model files exist
    model_file_mapping = {}
    training_mode = None
    
    # Check for upper bound (full training) - single pretrain_best_model.pth
    upperbound_model_path = args.model_dir if 'pretrain_best_model.pth' in args.model_dir else os.path.join(args.model_dir, 'pretrain_best_model.pth')

    if os.path.exists(upperbound_model_path):
        # Upper bound training - one model for all checkpoints
        training_mode = "upper_bound"
        for ckp in ckp_list:
            model_file_mapping[ckp] = upperbound_model_path
        log_success("Detected UPPER BOUND training mode")
        log_info(f"Using single model file: {upperbound_model_path}", Colors.GREEN)
        log_info(f"Will evaluate this model on all {len(ckp_list)} checkpoints: {ckp_list}", Colors.GREEN)
    else:
        # Check for accumulative training files (ckp_X_best_model.pth)
        for ckp in ckp_list:
            if ckp == 'ckp_1':
                # For ckp_1, use the original pretrained model (zero-shot)
                model_file_mapping[ckp] = 'pretrained_model'  # Special marker for pretrained model
                if training_mode is None:
                    training_mode = "accumulative"
                log_info(f"Using original pretrained model for {ckp} (zero-shot)", Colors.BLUE)
            else:
                acc_model_path = os.path.join(args.model_dir, f'ckp_{ckp}_best_model.pth')
                if os.path.exists(acc_model_path):
                    model_file_mapping[ckp] = acc_model_path
                    if training_mode is None:
                        training_mode = "accumulative"
                else:
                    log_warning(f"No model file found for {ckp}: {acc_model_path}")
        
        if training_mode == "accumulative":
            log_success("Detected ACCUMULATIVE training mode")
            log_info(f"Found model files for {len(model_file_mapping)} checkpoints", Colors.BLUE)
    
    if not model_file_mapping:
        error_msg = f"No model files found in {args.model_dir}. Expected either:\n"
        error_msg += f"  - Upper bound training: pretrain_best_model.pth directly in {args.model_dir}\n"
        error_msg += f"  - Accumulative training: ckp_X_best_model.pth files directly in {args.model_dir}"
        log_error(error_msg)
        raise ValueError(error_msg)
    
    # Filter to only checkpoints that have model files
    ckp_list = list(model_file_mapping.keys())
    
    # Create final summary
    mode_summary = f"Training mode: {training_mode}\n"
    mode_summary += f"Checkpoints to evaluate: {len(ckp_list)}\n"
    mode_summary += f"Checkpoint list: {', '.join(ckp_list)}"
    logging.info(create_info_box("Evaluation Plan", mode_summary))
    
    # Create output directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        log_success(f"Created output directory: {args.save_dir}")
    else:
        log_info(f"Using existing output directory: {args.save_dir}", Colors.CYAN)
    
    log_section_start("🎯 CHECKPOINT EVALUATION", Colors.BRIGHT_MAGENTA)
    # Run evaluation for each checkpoint
    eval_results = {}
    
    # Save zs backbone for interpolation if needed
    if args.interpolation_model or args.interpolation_head:
        _classifier = copy.deepcopy(classifier)
        _classifier = _classifier.to(args.device)

    preds = {}
    for i, ckp in enumerate(ckp_list):
        log_subsection_start(f"📊 Evaluating Checkpoint {ckp} ({i+1}/{len(ckp_list)})", Colors.CYAN)
        
        log_step(1, f"Loading model weights", Colors.BLUE)
        # Load model weights for this checkpoint
        model_path = model_file_mapping[ckp]

        preds[ckp] = {}
        
        try:
            if model_path == 'pretrained_model':
                # For ckp_1, use the original pretrained model (already loaded in classifier)
                log_info(f"Using original pretrained model for {ckp} (zero-shot)", Colors.GREEN)
                # classifier is already the pretrained model, no need to load anything
                classifier.to(args.device)
                classifier.eval()
            else:
                log_info(f"Loading model from {model_path}", Colors.GREEN)
                state_dict = torch.load(model_path, map_location=args.device)
                log_success(f"Loaded model weights from {model_path}")
                # Handle expanded head case with shape mismatch protection
                current_state_dict = classifier.state_dict()
                    
                # Filter out head parameters that have shape mismatches
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key in current_state_dict:
                        current_shape = current_state_dict[key].shape
                        saved_shape = value.shape
                        
                        if current_shape == saved_shape:
                            # Shapes match, safe to load
                            filtered_state_dict[key] = value
                        elif 'head' in key:
                            # Head parameter with shape mismatch - handle carefully
                            if key == 'head.weight':
                                # Copy only the overlapping classes
                                min_classes = min(current_shape[0], saved_shape[0])
                                current_state_dict[key][:min_classes] = value[:min_classes]
                                log_info(f"Copied {min_classes} class weights for head.weight", Colors.YELLOW)
                            elif key == 'head.bias':
                                # Copy only the overlapping classes
                                min_classes = min(current_shape[0], saved_shape[0])
                                current_state_dict[key][:min_classes] = value[:min_classes]
                                log_info(f"Copied {min_classes} class biases for head.bias", Colors.YELLOW)
                            else:
                                log_warning(f"Skipping head parameter {key} due to shape mismatch: {saved_shape} vs {current_shape}")
                        else:
                            log_warning(f"Skipping parameter {key} due to shape mismatch: {saved_shape} vs {current_shape}")
                    else:
                        log_warning(f"Unexpected key in checkpoint: {key}")
                
                # Load the filtered state dict
                missing_keys, unexpected_keys = classifier.load_state_dict(filtered_state_dict, strict=False)
                
                # Log what happened during loading
                if missing_keys:
                    head_missing = [k for k in missing_keys if 'head' in k]
                    other_missing = [k for k in missing_keys if 'head' not in k]
                    
                    if head_missing:
                        log_info(f"Expected missing head parameters for expanded classes: {len(head_missing)} keys", Colors.YELLOW)
                    if other_missing:
                        log_warning(f"Unexpected missing parameters: {other_missing}")
                    
                if unexpected_keys:
                    log_warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
                    
                log_success("Model loaded successfully with expanded head handling")
                classifier.to(args.device)
                classifier.eval()

            if (args.interpolation_model or args.interpolation_head) and _classifier is not None:
                logging.info(f"Interpolating model at checkpoint {ckp} with alpha {args.interpolation_alpha}")
                classifier.interpolate_model(_classifier, alpha=args.interpolation_alpha)
                
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("model_load", f"after_load_{ckp}")
            
            log_success("Model loaded successfully")
                
        except Exception as e:
            log_error(f"Failed to load model for {ckp}: {e}")
            continue
        
        if args.accu_eval:
            eval_results[ckp] = {}
            
            for idx in range(i, len(ckp_list)):
                log_step(2, f"Preparing evaluation dataset", Colors.YELLOW)
                accu_ckp = ckp_list[idx]
                log_info(f"Accumulating evaluation for checkpoint {accu_ckp}", Colors.YELLOW)
                ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=accu_ckp)
                log_info(f"Evaluation dataset size for {accu_ckp}: {len(ckp_eval_dset)}", Colors.CYAN)
                
                if len(ckp_eval_dset) == 0:
                    log_warning(f"No evaluation data found for checkpoint {accu_ckp}")
                    continue
                
                # Create data loader
                eval_loader = DataLoader(
                    ckp_eval_dset, 
                    batch_size=common_config['eval_batch_size'], 
                    shuffle=False, 
                    num_workers=4
                )
                log_success("Data loader created successfully")
                
                log_step(3, f"Running model evaluation", Colors.GREEN)
                # Run evaluation
                if args.gpu_memory_monitor:
                    gpu_monitor.log_memory_usage("evaluation", f"before_{accu_ckp}")
                    
                loss_arr, preds_arr, labels_arr, pred_true, pred_false = eval(
                    classifier, 
                    eval_loader, 
                    args.device, 
                    chop_head=common_config['chop_head']
                )
                
                preds[ckp][accu_ckp] = {
                    'preds_true': pred_true,
                    'preds_false': pred_false
                }

                if args.gpu_memory_monitor:
                    gpu_monitor.log_memory_usage("evaluation", f"after_{accu_ckp}")
                
                acc, balanced_acc = print_metrics(
                    loss_arr, 
                    preds_arr, 
                    labels_arr, 
                    len(class_names),
                    log_predix=f"📊 {ckp}: "
                )
                eval_loss = np.mean(loss_arr)
            
                # Log metrics with colors
                log_metric(f"Accuracy for ckp {accu_ckp} using ckp model {ckp}", acc, ".4f", Colors.BRIGHT_GREEN)
                log_metric(f"Balanced Accuracy for ckp {accu_ckp} using ckp model {ckp}", balanced_acc, ".4f", Colors.BRIGHT_GREEN)
                log_metric(f"Evaluation Loss for ckp {accu_ckp} using ckp model {ckp}", eval_loss, ".4f", Colors.BRIGHT_BLUE)

                # Store results (convert numpy types to Python types for JSON serialization)
                eval_results[ckp][accu_ckp] = {
                    'accuracy': float(acc),
                    'balanced_accuracy': float(balanced_acc),
                    'loss': float(eval_loss),
                    'num_samples': int(len(ckp_eval_dset))
                }

                # Weight sanity: compare per-ckp on its own eval set
                if weight_sanity_enabled and accu_ckp == ckp:
                    sanity_total += 1
                    expected = sanity_expected.get(ckp)
                    if expected is None:
                        log_warning(f"[Sanity] No reference balanced_accuracy for {ckp} in final_training_summary.json")
                    else:
                        diff = abs(float(balanced_acc) - float(expected))
                        if diff <= sanity_offset:
                            log_success(f"[Sanity] {ckp} balanced_accuracy matches within {sanity_offset:.2f} (diff={diff:.4f})")
                        else:
                            log_warning(f"[Sanity] {ckp} mismatch: eval={balanced_acc:.4f}, ref={expected:.4f}, diff={diff:.4f} > {sanity_offset:.2f}")
                            sanity_mismatches.append({
                                'checkpoint': ckp,
                                'eval_balanced_accuracy': float(balanced_acc),
                                'ref_balanced_accuracy': float(expected),
                                'diff': float(diff)
                            })

                # Compare with previous checkpoints that were also evaluated on the same data (accu_ckp)
                # Only compare if we're evaluating on the current checkpoint data (accu_ckp == ckp)
                if accu_ckp == ckp:
                    for pre in range(0, i):
                        pre_ckp = ckp_list[pre]
                        # Check if previous checkpoint was evaluated on this same data
                        if pre_ckp in eval_results and ckp in eval_results[pre_ckp]:
                            prev_balanced_acc = eval_results[pre_ckp][ckp]['balanced_accuracy']
                            difference = balanced_acc - prev_balanced_acc
                            eval_results[ckp][ckp]['improvement_over_' + pre_ckp] = difference
                                
                            if difference > 0:
                                log_info(f"🎯 Model {ckp} improved over {pre_ckp} by {difference:.4f} on {ckp} data!", Colors.CYAN)
                            else:
                                log_info(f"📊 Model {ckp} did not improve over {pre_ckp} on {ckp} data (Δ={difference:+.4f})", Colors.CYAN)

                log_step(5, f"Saving results", Colors.BLUE)
                # Save predictions if requested
                if args.save_predictions:
                    pred_path = os.path.join(args.save_dir, f'{ckp}_eval_preds.pkl')
                    with open(pred_path, 'wb') as f:
                        pickle.dump((preds_arr, labels_arr), f)
                    log_success(f"Predictions saved to {pred_path}")
                
                # Clear GPU cache between checkpoints
                if args.gpu_memory_monitor:
                    gpu_monitor.clear_cache_and_log(f"eval_checkpoint_{ckp}")
                
                log_subsection_start(f"✅ Checkpoint {ckp} Complete", Colors.BRIGHT_GREEN)
        else:
            log_step(2, f"Preparing evaluation dataset", Colors.YELLOW)
            # Get evaluation dataset for this checkpoint
            ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp)
            log_info(f"Evaluation dataset size for {ckp}: {len(ckp_eval_dset)}", Colors.CYAN)
            
            if len(ckp_eval_dset) == 0:
                log_warning(f"No evaluation data found for checkpoint {ckp}")
                continue
            
            # Create data loader
            eval_loader = DataLoader(
                ckp_eval_dset, 
                batch_size=common_config['eval_batch_size'], 
                shuffle=False, 
                num_workers=4
            )
            log_success("Data loader created successfully")
            
            log_step(3, f"Running model evaluation", Colors.GREEN)
            # Run evaluation
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"before_{ckp}")
                
            loss_arr, preds_arr, labels_arr, pred_true, pred_false = eval(
                classifier, 
                eval_loader, 
                args.device, 
                chop_head=common_config['chop_head']
            )

            preds[ckp] = {
                'preds_true': pred_true,
                'preds_false': pred_false
            }

            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"after_{ckp}")
            
            log_success("Model evaluation completed")
            
            log_step(4, f"Computing metrics", Colors.MAGENTA)
            # Calculate and log metrics
            acc, balanced_acc = print_metrics(
                loss_arr, 
                preds_arr, 
                labels_arr, 
                len(class_names),
                log_predix=f"📊 {ckp}: "
            )
            eval_loss = np.mean(loss_arr)
            
            # Log metrics with colors
            log_metric("Accuracy", acc, ".4f", Colors.BRIGHT_GREEN)
            log_metric("Balanced Accuracy", balanced_acc, ".4f", Colors.BRIGHT_GREEN)
            log_metric("Evaluation Loss", eval_loss, ".4f", Colors.BRIGHT_BLUE)
            
            # Store results (convert numpy types to Python types for JSON serialization)
            eval_results[ckp] = {
                'accuracy': float(acc),
                'balanced_accuracy': float(balanced_acc),
                'loss': float(eval_loss),
                'num_samples': int(len(ckp_eval_dset))
            }
            
            # Calibration
            if args.calibration and ckp != 'ckp_1':
                # Parse the number from ckp and create train_ckp list up to one before
                ckp_num = int(ckp.split('_')[1])
                train_ckp = [f"ckp_{i}" for i in range(1, ckp_num)]
                log_info(f"[CALIBRATION] Calibrating model trained with checkpoints: {train_ckp}", Colors.CYAN)
                
                ckp_train_dset = train_dset.get_subset(is_train=True, ckp_list=train_ckp)
                
                # Create data loader
                train_loader = DataLoader(
                    ckp_train_dset, 
                    batch_size=common_config['train_batch_size'],  
                    shuffle=False, 
                    num_workers=4
                )
                seen_classes = checkpoint_analysis[ckp]['seen'] if checkpoint_analysis and ckp in checkpoint_analysis else None
                unseen_classes = checkpoint_analysis[ckp]['unseen'] if checkpoint_analysis and ckp in checkpoint_analysis else None
                
                calibration_factor = run_alg_calibration(ckp_train_dset,
                                    train_loader,
                                    classifier,
                                    seen_classes,
                                    unseen_classes,
                                    args.device)
                log_success(f"Calibration Factor Retrieval Done: {calibration_factor:.4f}")
                
                unseen_classes = torch.tensor([ckp_train_dset.class_name_idx[name] 
                                               for name in unseen_classes], 
                                              dtype=torch.long) if unseen_classes else torch.tensor([], dtype=torch.long)
                loss_arr, preds_arr, labels_arr, pred_true, pred_false = eval(
                    classifier, 
                    eval_loader, 
                    args.device, 
                    chop_head=common_config['chop_head'],
                    calibration_factor=calibration_factor,
                    unseen_classes=unseen_classes
                )

                preds[ckp] = {
                    'preds_true': pred_true,
                    'preds_false': pred_false
                }

                log_success("Calibrated Model evaluation completed")

                # Calculate and log metrics
                acc, balanced_acc = print_metrics(
                    loss_arr, 
                    preds_arr, 
                    labels_arr, 
                    len(class_names),
                    log_predix=f"📊 {ckp}: "
                )
                eval_loss = np.mean(loss_arr)
                
                # Log metrics with colors
                log_metric("Accuracy", acc, ".4f", Colors.BRIGHT_GREEN)
                log_metric("Balanced Accuracy", balanced_acc, ".4f", Colors.BRIGHT_GREEN)
                log_metric("Evaluation Loss", eval_loss, ".4f", Colors.BRIGHT_BLUE)
                
                # Store results (convert numpy types to Python types for JSON serialization)
                eval_results[ckp] = {
                    'accuracy': float(acc),
                    'balanced_accuracy': float(balanced_acc),
                    'loss': float(eval_loss),
                    'num_samples': int(len(ckp_eval_dset))
                }

            # Weight sanity: compare per-ckp on its own eval set
            if weight_sanity_enabled:
                sanity_total += 1
                expected = sanity_expected.get(ckp)
                if expected is None:
                    log_warning(f"[Sanity] No reference balanced_accuracy for {ckp} in final_training_summary.json")
                else:
                    diff = abs(float(balanced_acc) - float(expected))
                    if diff <= sanity_offset:
                        log_success(f"[Sanity] {ckp} balanced_accuracy matches within {sanity_offset:.2f} (diff={diff:.4f})")
                    else:
                        log_warning(f"[Sanity] {ckp} mismatch: eval={balanced_acc:.4f}, ref={expected:.4f}, diff={diff:.4f} > {sanity_offset:.2f}")
                        sanity_mismatches.append({
                            'checkpoint': ckp,
                            'eval_balanced_accuracy': float(balanced_acc),
                            'ref_balanced_accuracy': float(expected),
                            'diff': float(diff)
                        })
            
            # Log to wandb if enabled
            if args.wandb:
                log_info(f"📈 Logging metrics to W&B: {ckp}", Colors.CYAN)
                wandb.log({
                    "eval/accuracy": acc,
                    "eval/balanced_accuracy": balanced_acc,
                    "eval/loss": eval_loss,
                    "eval/checkpoint": ckp,
                    "eval/num_samples": len(ckp_eval_dset)
                }, step=i)
            
            log_step(5, f"Saving results", Colors.BLUE)
            # Save predictions if requested
            if args.save_predictions:
                pred_path = os.path.join(args.save_dir, f'{ckp}_eval_preds.pkl')
                with open(pred_path, 'wb') as f:
                    pickle.dump((preds_arr, labels_arr), f)
                log_success(f"Predictions saved to {pred_path}")
            
            # Clear GPU cache between checkpoints
            if args.gpu_memory_monitor:
                gpu_monitor.clear_cache_and_log(f"eval_checkpoint_{ckp}")
            
            log_subsection_start(f"✅ Checkpoint {ckp} Complete", Colors.BRIGHT_GREEN)

    log_section_start("📊 FINAL EVALUATION SUMMARY", Colors.BRIGHT_YELLOW)
    
    # Calculate average metrics
    if eval_results:
        if not args.accu_eval:
            # Filter out NaN values when calculating averages
            valid_accuracies = [result['accuracy'] for result in eval_results.values() if not np.isnan(result['accuracy'])]
            valid_balanced_accuracies = [result['balanced_accuracy'] for result in eval_results.values() if not np.isnan(result['balanced_accuracy'])]
            valid_losses = [result['loss'] for result in eval_results.values() if not np.isnan(result['loss'])]
            
            avg_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0.0
            avg_balanced_accuracy = sum(valid_balanced_accuracies) / len(valid_balanced_accuracies) if valid_balanced_accuracies else 0.0
            avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0.0
            total_samples = sum(result['num_samples'] for result in eval_results.values())
        else:
            # For eval_accu mode, average over all accumulated results
            all_accuracies = []
            all_balanced_accuracies = []
            all_losses = []
            total_samples = 0
            for ckp, sub_results in eval_results.items():
                all_accuracies.append(sub_results[ckp]['accuracy'])
                all_balanced_accuracies.append(sub_results[ckp]['balanced_accuracy'])
                all_losses.append(sub_results[ckp]['loss'])
                total_samples += sub_results[ckp]['num_samples']
            avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
            avg_balanced_accuracy = sum(all_balanced_accuracies) / len(all_balanced_accuracies) if all_balanced_accuracies else 0.0
            avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
    else:
        avg_accuracy = avg_balanced_accuracy = avg_loss = 0.0
        total_samples = 0
    
    # Log summary with beautiful formatting
    logging.info("=== CHECKPOINT RESULTS SUMMARY ===")
    for ckp, results in eval_results.items():
        if args.accu_eval:
            logging.info(f"{ckp}: acc={results[ckp]['accuracy']:.4f}, balanced_acc={results[ckp]['balanced_accuracy']:.4f}, "
                    f"loss={results[ckp]['loss']:.4f}, samples={results[ckp]['num_samples']}")
        else:
            logging.info(f"{ckp}: acc={results['accuracy']:.4f}, balanced_acc={results['balanced_accuracy']:.4f}, "
                    f"loss={results['loss']:.4f}, samples={results['num_samples']}")
    logging.info("=" * 60)
    logging.info(f"🎯 FINAL AVERAGE ACROSS ALL CHECKPOINTS:")
    log_metric("Average Accuracy", avg_accuracy, ".4f", Colors.BRIGHT_GREEN)
    log_metric("Average Balanced Accuracy", avg_balanced_accuracy, ".4f", Colors.BRIGHT_GREEN)
    log_metric("Average Loss", avg_loss, ".4f", Colors.BRIGHT_BLUE)
    logging.info(f"  📊 Total Checkpoints: {len(eval_results)}")
    logging.info(f"  📊 Total Samples: {total_samples}")
    logging.info("=" * 60)

    # Final weight sanity summary
    if weight_sanity_enabled:
        log_section_start("🧪 WEIGHT SANITY CHECK SUMMARY", Colors.BRIGHT_CYAN)
        mism_cnt = len(sanity_mismatches)
        ok_cnt = max(0, sanity_total - mism_cnt)
        status = "PASS" if mism_cnt == 0 else "FAIL"
        log_info(f"Offset: ±{sanity_offset:.2f}")
        log_info(f"Compared checkpoints: {sanity_total}")
        log_info(f"Matches: {ok_cnt} | Mismatches: {mism_cnt}")
        if mism_cnt > 0:
            for item in sanity_mismatches:
                logging.info(f" - {item['checkpoint']}: eval={item['eval_balanced_accuracy']:.4f}, ref={item['ref_balanced_accuracy']:.4f}, diff={item['diff']:.4f}")
        log_final_result(f"Weight Sanity: {status}")
    
    eval_results['average'] = {}
    eval_results['average']['accuracy'] = float(avg_accuracy)
    eval_results['average']['balanced_accuracy'] = float(avg_balanced_accuracy)
    eval_results['average']['loss'] = float(avg_loss)

    # Save summary results
    summary_path = os.path.join(args.save_dir, 'eval_only_summary.json')
    preds_path = os.path.join(args.save_dir, 'eval_only_predictions.json')
    with open(summary_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    with open(preds_path, 'w') as f:
        json.dump(preds, f, indent=2)
    log_success(f"Summary saved to {summary_path}")

    # Final GPU memory summary
    if args.gpu_memory_monitor:
        summary = gpu_monitor.get_memory_summary()
        log_info(f"🔧 GPU Memory Summary: {summary}", Colors.CYAN)
        if args.wandb:
            wandb.log({"gpu_memory/summary": summary})
    
    # Finalize wandb if enabled
    if args.wandb:
        log_info(f"📈 Metrics logged to W&B project: Camera Trap Benchmark - EVAL ONLY", Colors.CYAN)
        wandb.finish()
        log_success("✅ W&B run finished")
    
    # ALG Calibration 
    # if args.calibration and not args.accu_eval:
    #     log_section_start("📊 CALIBRATION AUC CURVE SUMMARY", Colors.BRIGHT_MAGENTA)
        
    #     # Train Dataset
        
        
    #     # Seen / Unseen
    #     # Class Counts
       

    # Final completion logs
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    log_final_result("🎉 EVALUATION-ONLY MODE COMPLETED SUCCESSFULLY!")
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    
    # Final summary with colors
    log_info(f"📊 Total checkpoints evaluated: {len(eval_results)}")
    log_info(f"💾 Results saved to: {args.save_dir}")
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    print()

def parse_args():
    """Parse command-line arguments for the adaptive learning pipeline.
        
        Returns:
            args (argparse.Namespace): Parsed command-line arguments.
    """
    # Configurations
    parser = argparse.ArgumentParser(description='Adaptive Workflow')
    parser.add_argument('--c', type=str, help='Configuration file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--seed', type=int, default=9527, help='Random seed')
    parser.add_argument('--eval_per_epoch', action='store_true', help='Evaluate per epoch')
    parser.add_argument('--test_per_epoch', action='store_true', help='Test with current and next checkpoint test data after each epoch (requires eval_per_epoch)')
    parser.add_argument('--save_best_model', action='store_true', default=True, help='Save best model during training')
    parser.add_argument('--validation_mode', type=str, default='balanced_acc', choices=['balanced_acc', 'loss'], help='Metric to use for best model selection: balanced_acc (higher is better) or loss (lower is better)')
    parser.add_argument('--early_stop_epoch', type=int, default=5, help='Number of epochs without improvement to trigger early stopping after warmup period (default: 5)')
    parser.add_argument('--is_save', action='store_true', help='Save model')
    parser.add_argument('--eval_only', action='store_true', help='Evaluate only mode - loads trained model checkpoints and evaluates them')
    parser.add_argument('--eval_accu', action='store_true', help='Perform calibration analysis during eval_only mode')
    parser.add_argument('--model_dir', type=str, help='Directory containing trained model checkpoints (.pth files) for eval_only mode')
    parser.add_argument('--weight_sanity', action='store_true', help="Compare eval-only balanced_accuracy to reference in model_dir/final_training_summary.json with ±0.03 tolerance")
    parser.add_argument('--expand_head', type=str, default=None, help='Expand classifier head to include unseen classes during eval_only mode')
    parser.add_argument('--plot_features', action='store_true', help='Plot feature distributions')
    parser.add_argument('--plot_text_F', action='store_true', help='Plot text features')
    parser.add_argument('--checkpoint_list', type=str, nargs='+', default=None, help='Specific checkpoint numbers to evaluate (for eval_only mode). If not provided, evaluates all available checkpoints')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction arrays for each checkpoint (for eval_only mode)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')  # New argument
    parser.add_argument('--gpu_memory_monitor', action='store_true', help='Enable GPU memory monitoring and logging')  # New argument
    parser.add_argument('--no_gpu_monitor_colors', action='store_true', help='Disable colored output for GPU monitoring (colors enabled by default)')  # New argument
    parser.add_argument('--interpolation_model', action='store_true', help='Enable interpolation model')
    parser.add_argument('--interpolation_head', action='store_true', help='Enable interpolation head')
    parser.add_argument('--interpolation_alpha', type=float, default=0.5, help='Interpolation alpha value (default: 0.5)')
    parser.add_argument('--label_type', type=str, default='common', choices=['scientific', 'common'], help='Type of class names to use (scientific or common names)')

    ###########################Model Configurations#########################
    parser.add_argument('--pretrained_weights', type=str, default='bioclip2',
                        choices=['bioclip', 'bioclip2'],
                        help='pretrained weights name')

    parser.add_argument('--class_type', type=str, default='common_name',
                        choices=['common_name', 'scientific_name'],
                        help='Class type for the model')
    
    parser.add_argument('--drop_path_rate', default=0.,
                        type=float,
                        help='Drop Path Rate (default: %(default)s)')
    # parser.add_argument('--model', type=str, default='vit', choices=['vit', 'swin'],
    #                     help='pretrained model name')

    ############################## TEST #################################
    parser.add_argument('--accu_eval', action='store_true',
                        help='whether to test all later checkpoints after training on each checkpoint')

    ############################## Text Encoder ##############################
    parser.add_argument('--text', type=str, default='head',
                        choices=['head', 'full', 'lora'],
                        help='text encoder type, head for head only, full for full text encoder')
    parser.add_argument('--text_template', type=str, default='openai',
                        choices=['bioclip', 'openai', 'customized'],
                        help='text template type')

    ########################PETL#########################
    parser.add_argument('--ft_attn_module', default=None, choices=['adapter', 'convpass', 'repadapter'],
                        help='Module used to fine-tune attention module. (default: %(default)s)')
    parser.add_argument('--ft_attn_mode', default='parallel',
                        choices=['parallel', 'sequential_after', 'sequential_before'],
                        help='fine-tune mode for attention module. (default: %(default)s)')
    parser.add_argument('--ft_attn_ln', default='before',
                        choices=['before', 'after'],
                        help='fine-tune mode for attention module before layer norm or after. (default: %(default)s)')

    parser.add_argument('--ft_mlp_module', default=None, choices=['adapter', 'convpass', 'repadapter'],
                        help='Module used to fine-tune mlp module. (default: %(default)s)')
    parser.add_argument('--ft_mlp_mode', default='parallel',
                        choices=['parallel', 'sequential_after', 'sequential_before'],
                        help='fine-tune mode for mlp module. (default: %(default)s)')
    parser.add_argument('--ft_mlp_ln', default='before',
                        choices=['before', 'after'],
                        help='fine-tune mode for attention module before layer norm or after. (default: %(default)s)')

    ########################AdaptFormer/Adapter#########################
    parser.add_argument('--adapter_bottleneck', type=int, default=64,
                        help='adaptformer bottleneck middle dimension. (default: %(default)s)')
    parser.add_argument('--adapter_init', type=str, default='lora_kaiming',
                        choices=['lora_kaiming', 'xavier', 'zero', 'lora_xavier'],
                        help='how adapter is initialized')
    parser.add_argument('--adapter_scaler', default=0.1,
                        help='adaptformer scaler. (default: %(default)s)')

    ########################ConvPass#########################
    parser.add_argument('--convpass_xavier_init', action='store_true',
                        help='whether apply xavier_init to the convolution layer in ConvPass')
    parser.add_argument('--convpass_bottleneck', type=int, default=8,
                        help='convpass bottleneck middle dimension. (default: %(default)s)')
    parser.add_argument('--convpass_init', type=str, default='lora_xavier',
                        choices=['lora_kaiming', 'xavier', 'zero', 'lora_xavier'],
                        help='how convpass is initialized')
    parser.add_argument('--convpass_scaler', default=10, type=float,
                        help='ConvPass scaler. (default: %(default)s)')

    ########################VPT#########################
    parser.add_argument('--vpt_mode', type=str, default=None, choices=['deep', 'shallow'],
                        help='VPT mode, deep or shallow')
    parser.add_argument('--vpt_num', default=10, type=int,
                        help='Number of prompts (default: %(default)s)')
    parser.add_argument('--vpt_layer', default=None, type=int,
                        help='Number of layers to add prompt, start from the last layer (default: %(default)s)')
    parser.add_argument('--vpt_dropout', default=0.1, type=float,
                        help='VPT dropout rate for deep mode. (default: %(default)s)')

    ########################SSF#########################
    parser.add_argument('--ssf', action='store_true',
                        help='whether turn on Scale and Shift the deep Features (SSF) tuning')

    ########################lora_kaiming#########################
    parser.add_argument('--lora_bottleneck', type=int, default=0,
                        help='lora bottleneck middle dimension. (default: %(default)s)')

    ########################FacT#########################
    parser.add_argument('--fact_dim', type=int, default=8,
                        help='FacT dimension. (default: %(default)s)')
    parser.add_argument('--fact_type', type=str, default=None, choices=['tk', 'tt'],
                        help='FacT method')
    parser.add_argument('--fact_scaler', type=float, default=1.0,
                        help='FacT scaler. (default: %(default)s)')

    ########################repadapter#########################
    parser.add_argument('--repadapter_bottleneck', type=int, default=8,
                        help='repadapter bottleneck middle dimension. (default: %(default)s)')
    parser.add_argument('--repadapter_init', type=str, default='lora_xavier',
                        choices=['lora_xavier', 'lora_kaiming', 'xavier', 'zero'],
                        help='how repadapter is initialized')
    parser.add_argument('--repadapter_scaler', default=1, type=float,
                        help='repadapter scaler. (default: %(default)s)')
    parser.add_argument('--repadapter_group', type=int, default=2,
                        help='repadapter group')

    ########################BitFit#########################
    parser.add_argument('--bitfit', action='store_true',
                        help='whether turn on BitFit')

    ########################VQT#########################
    parser.add_argument('--vqt_num', default=0, type=int,
                        help='Number of query prompts (default: %(default)s)')
    parser.add_argument('--vqt_dropout', default=0.1, type=float,
                        help='VQT dropout rate for deep mode. (default: %(default)s)')

    ########################MLP#########################
    parser.add_argument('--mlp_index', default=None, type=int, nargs='+',
                        help='indexes of mlp to tune (default: %(default)s)')
    parser.add_argument('--mlp_type', type=str, default='full',
                        choices=['fc1', 'fc2', 'full'],
                        help='how mlps are tuned')

    ########################Attention#########################
    parser.add_argument('--attention_index', default=None, type=int, nargs='+',
                        help='indexes of attention to tune (default: %(default)s)')
    parser.add_argument('--attention_type', type=str, default='full',
                        choices=['qkv', 'proj', 'full'],
                        help='how attentions are tuned')

    ########################LayerNorm#########################
    parser.add_argument('--ln', action='store_true',
                        help='whether turn on LayerNorm fit')

    ########################DiffFit#########################
    parser.add_argument('--difffit', action='store_true',
                        help='whether turn on DiffFit')

    ########################full#########################
    parser.add_argument('--full', action='store_true',
                        help='whether turn on full finetune')
    ########################loss#########################
    # parser.add_argument('--loss', type=str, default='ce',
    #                     choices=['ce', 'focal', 'kd', 'cb', 'supcon', 'cdt'])

    ########################block#########################
    parser.add_argument('--block_index', default=None, type=int, nargs='+',
                        help='indexes of block to tune (default: %(default)s)')

    ########################domain generalization#########################
    parser.add_argument('--generalization_test', type=str, default='a',
                        choices=['v2', 's', 'a'],
                        help='domain generalization test set for imagenet')
    parser.add_argument('--merge_factor', default=1, type=float,
                        help='merge factor')
    parser.add_argument('--lora_interpolate', action='store_true',
                        help='whether to use LoRA interpolation')
    # parser.add_argument('--lora_interpolate', default=1, type=float,
    #                     help='shift factor')

    ########################calibration#########################
    parser.add_argument('--calibration', action='store_true', 
                        help='Enable paper calibration analysis for eval_only mode')

    args = parser.parse_args()

    # Override configurations
    if args.c:
        with open(args.c, 'r') as f:
            yml = yaml.YAML(typ='rt')
            config = yml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    args.gpu_id = None

    return args

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set PyTorch to deterministic mode for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional deterministic settings for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Force deterministic behavior for specific CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            torch.use_deterministic_algorithms(True)
        except:
            # Fallback for older PyTorch versions
            torch.backends.cudnn.deterministic = True
    
    # Set environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['WORKER_SEED'] = str(args.seed)  # Set seed for worker processes
    
    logging.info(f'Deterministic training enabled with seed: {args.seed}')

    save_dir = args.log_path
    if args.debug:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.save_dir = os.path.join(args.log_path, f"debug-{ts}")

    # Setup logging
    
    default_log_path = args.log_path
    
    args.save_dir = setup_logging(default_log_path, args.debug, args)
    logging.info(f'Saving to {args.save_dir}. ')

    # Save configuration
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yml = yaml.YAML()
        yml.dump(vars(args), f)

    # Run
    
    start_time = time.time()
    if args.eval_only:
        run_eval_only(args)
    else:
        run(args)
    end_time = time.time()

    # Print elapsed time with colors
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.1f}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"
    
    print(f"\n{Colors.BRIGHT_GREEN}⏱️  Total Execution Time: {Colors.BOLD}{time_str}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}🏁 All tasks completed successfully!{Colors.RESET}\n")

