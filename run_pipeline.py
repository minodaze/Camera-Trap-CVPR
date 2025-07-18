import argparse
import logging
import os
import copy
import time
import json
import re
import random

from datetime import datetime
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
from core.calibration import (
    comprehensive_paper_calibration, 
    get_training_classes_for_checkpoint,
    convert_numpy_types
)
from utils.misc import method_name
from utils.gpu_monitor import get_gpu_monitor, log_gpu_memory, monitor_model_memory
from utils.log_formatter import (
    setup_colored_logging, log_section_start, log_subsection_start, log_step,
    log_success, log_warning, log_error, log_info, log_config_item,
    log_checkpoint, log_final_result, log_metric, create_info_box, Colors,
    configure_colors_for_wandb
)

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
    log_path = os.path.join(log_path, petl_method_name)
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


def pretrain(classifier, class_names, pretrain_config, common_config, device, gpu_monitor=None, interpolation_model=False, interpolation_head=False, interpolation_alpha=0.5, eval_per_epoch=False, save_dir=None, args=None, test_per_epoch=False, eval_dset=None):
    """Pretrain the classifier on the pretraining dataset.
        
        Args:
            classifier (nn.Module): Classifier model.
            class_names (list): List of class names.
            pretrain_config (dict): Pretraining configuration.
            common_config (dict): Common configuration.
            device (str): Device to use for training.
            gpu_monitor: GPU memory monitor instance.
        Returns:
            classifier (nn.Module): Classifier after pretraining.
    
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
        
        # Log class distribution in validation set with beautiful styling
        val_class_counts = defaultdict(int)
        train_class_counts = defaultdict(int)
        for sample in val_samples:
            val_class_counts[sample.label] += 1
        for sample in train_samples:
            train_class_counts[sample.label] += 1
        
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
    return classifier

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
        match = re.search(r"logs/([^/]+)/([^/]+)/([^/]+)/", args.save_dir)
        wandb_run_name = "Train"
        if match:
            dataset = match.group(1)
            setting = match.group(2)
            # Construct the wandb run name
            wandb_run_name = f"Train | {dataset} | {setting}"

        module_name = getattr(args, 'module_name', 'default_module')  # Fallback if module_name is not in args
        wandb.init(
            project="Camera Trap Final Accum",  # Replace with your project name
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
    eval_dset = CkpDataset(common_config["eval_data_config_path"], class_names, label_type=label_type)
    
    # Monitor dataset memory usage if enabled
    if args.gpu_memory_monitor:
        gpu_monitor.log_memory_usage("dataset_load", "after_load", {
            'train_dataset_size': len(train_dset),
            'eval_dataset_size': len(eval_dset)
        })
    
    # Print ckp dict
    ckp_list = train_dset.get_ckp_list()
    
    dataset_summary = f"Training dataset size: {len(train_dset)}\n"
    dataset_summary += f"Evaluation dataset size: {len(eval_dset)}\n"
    dataset_summary += f"Number of checkpoints: {len(ckp_list)}\n"
    dataset_summary += f"Checkpoints: {', '.join(ckp_list)}"
    logging.info(create_info_box("Dataset Information", dataset_summary))
    
    log_section_start("🎯 PRETRAINING PHASE", Colors.BRIGHT_GREEN)
    
    # Pretrain
    if pretrain_config['pretrain']:
        log_info(f"Pretraining enabled with {pretrain_config['epochs']} epochs")
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("pretrain", "before")
        classifier = pretrain(classifier, 
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
                              eval_dset=eval_dset)

        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("pretrain", "after")
            gpu_monitor.clear_cache_and_log("pretrain")
        log_success("Pretraining completed successfully")
    else:
        log_warning("Pretraining skipped (disabled in configuration)")
    
    log_step(5, "Initializing pipeline modules")
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
    
    # Main loop
    for i in range(len(ckp_list)):
        # Get checkpoint
        ckp_prev = ckp_list[i - 1] if i > 0 else None
        ckp = ckp_list[i]
        
        log_subsection_start(f"📝 Processing Checkpoint {ckp} ({i+1}/{len(ckp_list)})", Colors.CYAN)
        
        # Monitor memory at checkpoint start
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("checkpoint", f"start_{ckp}")
        
        # Get training and evaluation dataset
        ckp_train_dset = train_dset.get_subset(is_train=True, ckp_list=ckp_prev)
        
        ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp)
        logging.info(f'Training dataset size: {len(ckp_train_dset)}. ')
        train_cls_count = {}
        for sample in ckp_train_dset.samples:
            label = sample.label
            train_cls_count[label] = train_cls_count.get(label, 0) + 1
        sum_count = sum(train_cls_count.values())
        for cls_idx, count in train_cls_count.items():
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            logging.info(f'  Class {cls_name} (idx {cls_idx}): {count} samples {count / sum_count:.2%}')

        logging.info(f'Evaluation dataset size: {len(ckp_eval_dset)}. ')

        eval_cls_count = {}
        for sample in ckp_eval_dset.samples:
            label = sample.label
            eval_cls_count[label] = eval_cls_count.get(label, 0) + 1
        sum_count = sum(eval_cls_count.values())
        for cls_idx, count in eval_cls_count.items():
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            logging.info(f'  Class {cls_name} (idx {cls_idx}): {count} samples {count / sum_count:.2%}')

        eval_cls_count = {}
        for sample in ckp_eval_dset.samples:
            label = sample.label
            eval_cls_count[label] = eval_cls_count.get(label, 0) + 1
        sum_count = sum(eval_cls_count.values())
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
        
        # Run OOD detection
        # Check if first checkpoint
        is_first_ckp = (i == 0)
        is_zs = (args.module_name == 'zs')
        log_step(1, f"Out-of-Distribution Detection ({ood_config.get('method', 'none')})", Colors.YELLOW)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("ood", f"before_{ckp}")
        classifier, ood_mask = ood_module.process(classifier, ckp_train_dset, ckp_eval_dset, train_dset_mask, is_first_ckp=is_first_ckp, is_zs=is_zs)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("ood", f"after_{ckp}")
        log_info(f"OOD samples identified: {ood_mask.sum()} / {len(ood_mask)}", Colors.YELLOW)
        
        # Run active learning
        log_step(2, f"Active Learning ({al_config.get('method', 'none')})", Colors.BLUE)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("active_learning", f"before_{ckp}")
        classifier, al_mask = al_module.process(
            classifier, 
            ckp_train_dset, 
            ckp_eval_dset, 
            ood_mask, 
            ckp=ckp
        )
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("active_learning", f"after_{ckp}")
        log_info(f"Active learning samples selected: {al_mask.sum()} / {len(al_mask)}", Colors.BLUE)

        # Prepare evaluation dataloader
        cl_eval_loader = DataLoader(ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False, worker_init_fn=worker_init_fn)

        # Prepare validation loader for continual learning
        cl_validation_loader = None
        cl_next_test_loader = None
        
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
        
        # Prepare test loader for test_per_epoch evaluation (next checkpoint only)
        if args.test_per_epoch and args.eval_per_epoch:
            cl_method = cl_config.get('method', 'none')
            if 'accumulative' in cl_method:
                # For accumulative: use current checkpoint test data as next test data
                cl_next_test_loader = cl_eval_loader  # This is current ckp test data
                log_info(f"Test per epoch: next test data from {ckp} (forward evaluation)", Colors.RED)
            else:
                # For non-accumulative: check if there's a next checkpoint
                if i + 1 < len(ckp_list):
                    next_ckp = ckp_list[i + 1]
                    next_ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=next_ckp)
                    cl_next_test_loader = DataLoader(next_ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False, worker_init_fn=worker_init_fn)
                    log_info(f"Test per epoch: next test data from {next_ckp}", Colors.RED)
                else:
                    cl_next_test_loader = None
                    log_info("Test per epoch: no next checkpoint available", Colors.YELLOW)
            
            # Set test loader in args for the continual learning module to access
            args._next_test_loader = cl_next_test_loader
        else:
            # Clear test loader if not using test_per_epoch
            args._next_test_loader = None

        # Run continual learning
        log_step(3, f"Continual Learning ({cl_config.get('method', 'none')})", Colors.GREEN)
        
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("continual_learning", f"before_{ckp}")
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

        log_checkpoint(f"Training completed for checkpoint {ckp}")
        
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
                loss_arr, preds_arr, labels_arr = eval(classifier, cl_eval_loader, args.device, chop_head=common_config['chop_head'])
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
                loss_arr, preds_arr, labels_arr = eval(classifier, current_cl_eval_loader, args.device, chop_head=common_config['chop_head'])
                print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"📊 Current ckp {ckp_prev}: ")
            
            log_info(f"Evaluating on target checkpoint {ckp}")
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"before_{ckp}")
            loss_arr, preds_arr, labels_arr = eval(classifier, cl_eval_loader, args.device, chop_head=common_config['chop_head'])
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"after_{ckp}")
            acc, balanced_acc = print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"📊 Target ckp {ckp}: ")
            eval_loss = np.mean(loss_arr)

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

        # Store final evaluation results for summary
        final_eval_results[ckp] = {
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'loss': float(eval_loss),
            'num_samples': len(preds_arr)
        }

        log_step(5, "Saving Results", Colors.MAGENTA)
        if args.is_save:
            save_path = os.path.join(args.save_dir, f'{ckp}.pth')
            torch.save(classifier.state_dict(), save_path)
            log_success(f"Model saved to {save_path}")
        
        pred_path = os.path.join(args.save_dir, f'{ckp}_preds.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump((preds_arr, labels_arr), f)
        log_success(f"Predictions saved to {pred_path}")
        
        mask_path = os.path.join(args.save_dir, f'{ckp}_mask.pkl')
        with open(mask_path, 'wb') as f:
            pickle.dump((ood_mask, al_mask), f)
        log_success(f"Masks saved to {mask_path}")
        
        # Clear GPU cache between checkpoints
        if args.gpu_memory_monitor:
            gpu_monitor.clear_cache_and_log(f"checkpoint_end_{ckp}")
        
        log_subsection_start(f"✅ Checkpoint {ckp} Complete", Colors.BRIGHT_GREEN)

    # Calculate and report final average balanced accuracy across all checkpoints
    if final_eval_results:
        # Calculate average metrics
        avg_accuracy = sum(result['accuracy'] for result in final_eval_results.values()) / len(final_eval_results)
        avg_balanced_accuracy = sum(result['balanced_accuracy'] for result in final_eval_results.values()) / len(final_eval_results)
        avg_loss = sum(result['loss'] for result in final_eval_results.values()) / len(final_eval_results)
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
        
        summary_path = os.path.join(args.save_dir, 'final_training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2)
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


def generate_calibration_auc_curve(test_logits, test_labels, train_logits, train_labels, 
                                  training_classes, class_names, ckp, save_dir):
    """
    Generate AUC curve showing the trade-off between seen class accuracy and absent class accuracy.
    
    This creates a curve similar to the paper's figure showing how calibration affects
    the balance between fine-tuning class accuracy and absent class accuracy.
    
    Args:
        test_logits: Test logits [N, C]
        test_labels: Test labels [N]
        train_logits: Training logits [N_train, C] (for learning gamma)
        train_labels: Training labels [N_train]
        training_classes: List of seen class indices
        class_names: List of all class names
        ckp: Checkpoint name
        save_dir: Directory to save the plot
    """
    try:
        from core.calibration import learn_gamma_alg, logit_bias_correction
        from sklearn.metrics import accuracy_score
        
        # Get absent classes
        num_classes = len(class_names)
        absent_classes = [i for i in range(num_classes) if i not in training_classes]
        
        # Debug information
        log_info(f"🔍 AUC Curve Debug for {ckp}:", Colors.CYAN)
        log_info(f"   Total classes: {num_classes}", Colors.CYAN)
        log_info(f"   Training classes: {len(training_classes)} -> {training_classes[:10]}{'...' if len(training_classes) > 10 else ''}", Colors.CYAN)
        log_info(f"   Absent classes: {len(absent_classes)} -> {absent_classes[:10]}{'...' if len(absent_classes) > 10 else ''}", Colors.CYAN)
        
        # Check class presence in test data
        unique_test_labels = np.unique(test_labels)
        log_info(f"   Test data classes: {len(unique_test_labels)} -> {unique_test_labels[:10].tolist()}{'...' if len(unique_test_labels) > 10 else ''}", Colors.CYAN)
        
        if not absent_classes or not training_classes:
            log_warning(f"Cannot generate AUC curve for {ckp}: missing seen or absent classes")
            log_warning(f"   Training classes empty: {len(training_classes) == 0}")
            log_warning(f"   Absent classes empty: {len(absent_classes) == 0}")
            return None
        
        # Separate test samples by class type
        seen_mask = np.isin(test_labels, training_classes)
        absent_mask = np.isin(test_labels, absent_classes)
        
        # More detailed debugging
        seen_count = np.sum(seen_mask)
        absent_count = np.sum(absent_mask)
        log_info(f"   Test samples with seen classes: {seen_count}/{len(test_labels)}", Colors.CYAN)
        log_info(f"   Test samples with absent classes: {absent_count}/{len(test_labels)}", Colors.CYAN)
        
        if not np.any(seen_mask) or not np.any(absent_mask):
            log_warning(f"Cannot generate full AUC curve for {ckp}: missing seen or absent test samples")
            log_warning(f"   Seen samples in test: {seen_count}")
            log_warning(f"   Absent samples in test: {absent_count}")
            
            # Show which training classes are actually in test data
            training_in_test = [cls for cls in training_classes if cls in unique_test_labels]
            absent_in_test = [cls for cls in absent_classes if cls in unique_test_labels]
            log_info(f"   Training classes found in test: {len(training_in_test)}/{len(training_classes)} -> {training_in_test[:10]}{'...' if len(training_in_test) > 10 else ''}", Colors.YELLOW)
            log_info(f"   Absent classes found in test: {len(absent_in_test)}/{len(absent_classes)} -> {absent_in_test[:10]}{'...' if len(absent_in_test) > 10 else ''}", Colors.YELLOW)
            
            # Try alternative approach: generate a simpler visualization
            if np.any(seen_mask) and not np.any(absent_mask):
                log_info(f"   Generating seen-classes-only calibration plot for {ckp}", Colors.BLUE)
                return generate_simple_calibration_plot(test_logits, test_labels, train_logits, train_labels,
                                                       training_classes, class_names, ckp, save_dir, "seen_only")
            elif np.any(absent_mask) and not np.any(seen_mask):
                log_info(f"   Generating absent-classes-only calibration plot for {ckp}", Colors.BLUE)
                return generate_simple_calibration_plot(test_logits, test_labels, train_logits, train_labels,
                                                       training_classes, class_names, ckp, save_dir, "absent_only")
            else:
                log_warning(f"   No compatible test samples found for {ckp} - skipping AUC curve")
                return None
        
        # Learn base gamma from training data
        if train_logits is not None and train_labels is not None:
            base_gamma = learn_gamma_alg(train_logits, training_classes)
        else:
            base_gamma = learn_gamma_alg(test_logits, training_classes)
        
        # Generate curve by varying gamma around the learned value
        gamma_range = np.linspace(base_gamma - 3.0, base_gamma + 3.0, 50)
        seen_accuracies = []
        absent_accuracies = []
        
        for gamma in gamma_range:
            # Apply calibration with this gamma
            corrected_logits = logit_bias_correction(test_logits, training_classes, gamma)
            corrected_preds = np.argmax(corrected_logits, axis=1)
            
            # Calculate accuracies for seen and absent classes
            seen_acc = accuracy_score(test_labels[seen_mask], corrected_preds[seen_mask])
            absent_acc = accuracy_score(test_labels[absent_mask], corrected_preds[absent_mask])
            
            seen_accuracies.append(seen_acc * 100)  # Convert to percentage
            absent_accuracies.append(absent_acc * 100)  # Convert to percentage
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot the curve
        plt.plot(seen_accuracies, absent_accuracies, 'r-', linewidth=2, label='Fine-tuning + γ')
        
        # Mark special points
        # Original point (gamma = 0)
        original_logits = test_logits.copy()
        original_preds = np.argmax(original_logits, axis=1)
        orig_seen_acc = accuracy_score(test_labels[seen_mask], original_preds[seen_mask]) * 100
        orig_absent_acc = accuracy_score(test_labels[absent_mask], original_preds[absent_mask]) * 100
        plt.plot(orig_seen_acc, orig_absent_acc, 'rs', markersize=10, label='Fine-tuning', zorder=5)
        
        # Optimal point (learned gamma)
        opt_corrected_logits = logit_bias_correction(test_logits, training_classes, base_gamma)
        opt_corrected_preds = np.argmax(opt_corrected_logits, axis=1)
        opt_seen_acc = accuracy_score(test_labels[seen_mask], opt_corrected_preds[seen_mask]) * 100
        opt_absent_acc = accuracy_score(test_labels[absent_mask], opt_corrected_preds[absent_mask]) * 100
        plt.plot(opt_seen_acc, opt_absent_acc, 'k*', markersize=15, label=f'{ckp} (γ={base_gamma:.2f})', zorder=5)
        
        # Pre-training point (theoretical - equal performance)
        if len(training_classes) > 0 and len(absent_classes) > 0:
            # Estimate pre-training performance as balanced point
            balanced_acc = (orig_seen_acc + orig_absent_acc) / 2
            plt.plot(balanced_acc, balanced_acc, 'g*', markersize=12, label='Pre-training (estimated)', zorder=5)
        
        # Styling to match the reference figure
        plt.xlabel('Fine-tuning Class Data Accuracy', fontsize=14)
        plt.ylabel('Absent Class Data Accuracy', fontsize=14)
        plt.title(f'{ckp}: Fine-tuning vs Absent Class Accuracy\n({len(training_classes)} seen, {len(absent_classes)} absent classes)', fontsize=16, fontweight='bold')
        
        # Set grid and limits
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(100, max(seen_accuracies) + 5))
        plt.ylim(0, max(100, max(absent_accuracies) + 5))
        
        # Add diagonal line for reference (equal performance)
        max_val = max(max(seen_accuracies), max(absent_accuracies))
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal performance')
        
        # Legend
        plt.legend(loc='best', fontsize=12)
        
        # Add text box with statistics
        stats_text = f"Training Classes: {len(training_classes)}\n"
        stats_text += f"Absent Classes: {len(absent_classes)}\n"
        stats_text += f"Test Samples: {len(test_labels)}\n"
        stats_text += f"Learned γ: {base_gamma:.3f}\n"
        stats_text += f"Seen Acc Δ: {opt_seen_acc - orig_seen_acc:+.1f}%\n"
        stats_text += f"Absent Acc Δ: {opt_absent_acc - orig_absent_acc:+.1f}%"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(save_dir, f'{ckp}_calibration_auc_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_success(f"📊 AUC curve saved to {plot_path}")
        
        # Return summary data
        return {
            'plot_path': plot_path,
            'original_seen_acc': orig_seen_acc,
            'original_absent_acc': orig_absent_acc,
            'calibrated_seen_acc': opt_seen_acc,
            'calibrated_absent_acc': opt_absent_acc,
            'gamma': base_gamma,
            'seen_improvement': opt_seen_acc - orig_seen_acc,
            'absent_improvement': opt_absent_acc - orig_absent_acc
        }
        
    except Exception as e:
        log_error(f"Failed to generate AUC curve for {ckp}: {e}")
        return None


def generate_simple_calibration_plot(test_logits, test_labels, train_logits, train_labels,
                                   training_classes, class_names, ckp, save_dir, plot_type="seen_only"):
    """
    Generate a simple calibration plot when full AUC curve cannot be generated.
    
    Args:
        test_logits: Test logits [N, C]
        test_labels: Test labels [N]
        train_logits: Training logits [N_train, C]
        train_labels: Training labels [N_train]
        training_classes: List of seen class indices
        class_names: List of all class names
        ckp: Checkpoint name
        save_dir: Directory to save the plot
        plot_type: "seen_only" or "absent_only"
    """
    try:
        from core.calibration import learn_gamma_alg, logit_bias_correction
        from sklearn.metrics import accuracy_score
        
        # Learn base gamma
        if train_logits is not None and train_labels is not None:
            base_gamma = learn_gamma_alg(train_logits, training_classes)
        else:
            base_gamma = learn_gamma_alg(test_logits, training_classes)
        
        # Generate gamma range
        gamma_range = np.linspace(base_gamma - 2.0, base_gamma + 2.0, 30)
        accuracies = []
        
        # Determine which classes to focus on
        if plot_type == "seen_only":
            focus_classes = training_classes
            title_suffix = "Seen Classes Only"
            ylabel = "Seen Class Accuracy (%)"
        else:  # absent_only
            focus_classes = [i for i in range(len(class_names)) if i not in training_classes]
            title_suffix = "Absent Classes Only"
            ylabel = "Absent Class Accuracy (%)"
        
        focus_mask = np.isin(test_labels, focus_classes)
        
        if not np.any(focus_mask):
            log_warning(f"No {plot_type} samples found - cannot generate simple plot")
            return None
        
        # Calculate accuracies for different gamma values
        for gamma in gamma_range:
            corrected_logits = logit_bias_correction(test_logits, training_classes, gamma)
            corrected_preds = np.argmax(corrected_logits, axis=1)
            acc = accuracy_score(test_labels[focus_mask], corrected_preds[focus_mask])
            accuracies.append(acc * 100)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the curve
        plt.plot(gamma_range, accuracies, 'b-', linewidth=2, label=f'{title_suffix} Accuracy')
        
        # Mark special points
        original_preds = np.argmax(test_logits, axis=1)
        orig_acc = accuracy_score(test_labels[focus_mask], original_preds[focus_mask]) * 100
        plt.plot(0, orig_acc, 'rs', markersize=10, label='No Calibration (γ=0)')
        
        # Optimal point
        opt_corrected_logits = logit_bias_correction(test_logits, training_classes, base_gamma)
        opt_corrected_preds = np.argmax(opt_corrected_logits, axis=1)
        opt_acc = accuracy_score(test_labels[focus_mask], opt_corrected_preds[focus_mask]) * 100
        plt.plot(base_gamma, opt_acc, 'k*', markersize=15, label=f'Optimal (γ={base_gamma:.2f})')
        
        # Styling
        plt.xlabel('Gamma (γ)', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(f'{ckp}: Calibration Effect - {title_suffix}\n({len(focus_classes)} classes, {np.sum(focus_mask)} test samples)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text box with statistics
        stats_text = f"Focus Classes: {len(focus_classes)}\n"
        stats_text += f"Test Samples: {np.sum(focus_mask)}\n"
        stats_text += f"Learned γ: {base_gamma:.3f}\n"
        stats_text += f"Accuracy Δ: {opt_acc - orig_acc:+.1f}%"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(save_dir, f'{ckp}_calibration_{plot_type}_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_success(f"📊 Simple calibration plot saved to {plot_path}")
        
        return {
            'plot_path': plot_path,
            'plot_type': plot_type,
            'original_acc': orig_acc,
            'calibrated_acc': opt_acc,
            'gamma': base_gamma,
            'improvement': opt_acc - orig_acc,
            'focus_classes': len(focus_classes),
            'test_samples': int(np.sum(focus_mask))
        }
        
    except Exception as e:
        log_error(f"Failed to generate simple calibration plot for {ckp}: {e}")
        return None


def get_training_logits(classifier, train_dset, ckp, device, batch_size=32):
    """
    Get training logits for calibration analysis.
    
    Args:
        classifier: Model to evaluate
        train_dset: Training dataset object
        ckp: Current checkpoint
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        tuple: (train_logits, train_labels) or (None, None) if no training data
    """
    try:
        if ckp == 'ckp_1':
            # Zero-shot model - no training data
            return None, None
        
        # For ckp_N, get training data from ckp_(N-1)
        ckp_num = int(ckp.split('_')[1])
        prev_ckp = f'ckp_{ckp_num - 1}'
        
        # Get training dataset for previous checkpoint
        prev_train_dset = train_dset.get_subset(is_train=True, ckp_list=prev_ckp)
        
        if len(prev_train_dset) == 0:
            log_warning(f"No training data found for {prev_ckp}")
            return None, None
        
        # Create data loader
        train_loader = DataLoader(
            prev_train_dset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Get logits using eval_with_logits function
        _, _, train_labels, train_logits = eval_with_logits(
            classifier, train_loader, device, chop_head=False
        )
        
        log_info(f"Collected training logits from {prev_ckp}: {len(train_labels)} samples", Colors.CYAN)
        return train_logits, train_labels
        
    except Exception as e:
        log_warning(f"Failed to get training logits for {ckp}: {e}")
        return None, None


def eval_with_logits(classifier, loader, device, chop_head=False):
    """
    Evaluate model and return logits, predictions, and labels.
    Modified version of the core eval function that also returns logits.
    
    Args:
        classifier: Model to evaluate
        loader: DataLoader for evaluation data
        device: Device to run evaluation on
        chop_head: Whether to remove classification head
    
    Returns:
        tuple: (loss_arr, preds_arr, labels_arr, logits_arr)
    """
    dset = loader.dataset
    if len(dset) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    classifier.eval()
    loss_arr = []
    preds_arr = []
    labels_arr = []
    logits_arr = []
    n_classes = len(dset.class_names)

    if chop_head:
        observed_classes = set()
        for sample in dset.samples:
            cls = sample.label
            observed_classes.add(cls)
        observed_classes = list(observed_classes)
        observed_classes.sort()
        chop_mask = np.ones(n_classes, dtype=bool)
        chop_mask[observed_classes] = 0
    else:
        chop_mask = np.zeros(n_classes, dtype=bool)
    
    with torch.no_grad():
        for inputs, labels, _, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if chop_head:
                _ = classifier(inputs)
            else:
                logits = classifier(inputs)
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                
                # Apply chop mask
                logits[:, chop_mask] = -np.inf
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Store results
                loss_arr.extend(loss.detach().cpu().numpy())
                preds_arr.extend(preds.detach().cpu().numpy())
                labels_arr.extend(labels.detach().cpu().numpy())
                logits_arr.extend(logits.detach().cpu().numpy())
    
    return np.array(loss_arr), np.array(preds_arr), np.array(labels_arr), np.array(logits_arr)


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
    
    log_step(3, "Loading data configuration")
    # Load configuration
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
    del data, data_test
    
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
    eval_dset = CkpDataset(common_config["eval_data_config_path"], class_names, is_train=False, label_type=label_type)

    # Get checkpoint list
    ckp_list = eval_dset.get_ckp_list()
    log_info(f"Available checkpoints in dataset: {ckp_list}", Colors.CYAN)
    
    # Simplified checkpoint selection for this implementation
    if args.checkpoint_list:
        target_checkpoints = [f"ckp_{ckp_str}" for ckp_str in args.checkpoint_list if f"ckp_{ckp_str}" in ckp_list]
        if target_checkpoints:
            ckp_list = target_checkpoints
            log_success(f"Will evaluate specified checkpoints: {ckp_list}")
        else:
            log_warning("No valid checkpoints found, using all available")
    
    # Create output directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        log_success(f"Created output directory: {args.save_dir}")
    
    log_section_start("🎯 CHECKPOINT EVALUATION", Colors.BRIGHT_MAGENTA)
    # Run evaluation for each checkpoint
    eval_results = {}
    
    for i, ckp in enumerate(ckp_list):
        log_subsection_start(f"📊 Evaluating Checkpoint {ckp} ({i+1}/{len(ckp_list)})", Colors.CYAN)
        
        log_step(1, f"Loading model weights", Colors.BLUE)
        # Load model weights for this checkpoint
        model_path = os.path.join(args.model_dir, f'ckp_{ckp}_best_model.pth')
        
        try:
            if os.path.exists(model_path):
                classifier.load_state_dict(torch.load(model_path, map_location=args.device))
                log_success(f"Loaded model weights from {model_path}")
            else:
                log_warning(f"Model file not found: {model_path}, using current model state")
        except Exception as e:
            log_error(f"Failed to load model for {ckp}: {e}")
            continue
        
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
        
        log_step(3, f"Running model evaluation", Colors.GREEN)
        # Run evaluation with logits for calibration
        if args.calibration:
            loss_arr, preds_arr, labels_arr, logits_arr = eval_with_logits(
                classifier, eval_loader, args.device, chop_head=common_config['chop_head']
            )
        else:
            loss_arr, preds_arr, labels_arr = eval(
                classifier, eval_loader, args.device, chop_head=common_config['chop_head']
            )
            logits_arr = None
        
        # Calculate and log metrics
        acc, balanced_acc = print_metrics(
            loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"📊 {ckp}: "
        )
        eval_loss = np.mean(loss_arr)
        
        # Initialize calibration results
        calibration_results = {}
        
        if args.calibration and logits_arr is not None:
            log_subsection_start("🎯 PAPER CALIBRATION ANALYSIS", Colors.BRIGHT_CYAN)
            
            # Determine training classes for paper's method
            if ckp == 'ckp_1':
                log_warning("ckp_1 uses zero-shot model - no training classes available")
                training_classes = []
            else:
                # For ckp_N, get training classes from ckp_(N-1) training data
                try:
                    train_dset = CkpDataset(common_config["train_data_config_path"], class_names, label_type=label_type)
                    training_classes = get_training_classes_for_checkpoint(ckp, train_dset, label_type)
                    log_info(f"📋 Training classes for {ckp}: {len(training_classes)} classes -> {training_classes[:10]}{'...' if len(training_classes) > 10 else ''}", Colors.GREEN)
                except Exception as e:
                    log_warning(f"Could not determine training classes: {e}")
                    training_classes = list(set(labels_arr.tolist()))
                    log_info(f"Fallback: using eval classes as training classes: {len(training_classes)} classes", Colors.YELLOW)
            
            # Show test data class distribution for comparison
            unique_test_classes = np.unique(labels_arr)
            log_info(f"📋 Test classes for {ckp}: {len(unique_test_classes)} classes -> {unique_test_classes[:10].tolist()}{'...' if len(unique_test_classes) > 10 else ''}", Colors.CYAN)
            
            # Get training logits for proper calibration
            log_step("4a", "Collecting training data for calibration", Colors.BLUE)
            try:
                train_dset = CkpDataset(common_config["train_data_config_path"], class_names, label_type=label_type)
                train_logits, train_labels = get_training_logits(classifier, train_dset, ckp, args.device, 
                                                               common_config.get('eval_batch_size', 32))
            except Exception as e:
                log_warning(f"Could not get training logits: {e}")
                train_logits, train_labels = None, None
            
            # Run paper calibration analysis
            log_step("4b", "Running paper calibration analysis", Colors.BLUE)
            
            calib_results = comprehensive_paper_calibration(
                logits_arr,  # test logits
                labels_arr,  # test labels
                class_names, 
                training_classes=training_classes,
                train_logits=train_logits,  # training logits (proper method)
                train_labels=train_labels,  # training labels (proper method)
                verbose=True
            )
            
            # Generate AUC curve for calibration analysis
            log_step("4c", "Generating calibration AUC curve", Colors.MAGENTA)
            auc_curve_data = generate_calibration_auc_curve(
                logits_arr, labels_arr, train_logits, train_labels,
                training_classes, class_names, ckp, args.save_dir
            )
            
            # Store calibration results
            calibration_results = {
                'checkpoint': ckp,
                'training_classes': training_classes,
                'original_accuracy': float(acc),
                'original_balanced_accuracy': float(balanced_acc),
                'auc_curve': auc_curve_data  # Add AUC curve data
            }
            
            # Add paper calibration results  
            if calib_results.get('paper_calibration') is not None:
                paper_results = calib_results['paper_calibration']
                calibration_results.update({
                    'paper_calibration': {
                        'gamma': float(paper_results['gamma']),
                        'accuracy': float(paper_results['corrected_accuracy']),
                        'balanced_accuracy': float(paper_results['balanced_accuracy']),
                        'accuracy_improvement': float(paper_results['accuracy_improvement']),
                        'balanced_accuracy_improvement': float(paper_results['balanced_accuracy_change']),
                        'absent_classes': paper_results['absent_classes'],
                        'absent_accuracy_improvement': float(paper_results['absent_accuracy_improvement']),
                        'seen_accuracy_improvement': float(paper_results['seen_accuracy_improvement'])
                    }
                })
                
                log_success(f"📄 Paper Method: γ={paper_results['gamma']:.3f}, Acc Δ={paper_results['accuracy_improvement']:+.4f}, Bal_Acc Δ={paper_results['balanced_accuracy_change']:+.4f}")
                log_success(f"   Absent Class Improvement: Δ={paper_results['absent_accuracy_improvement']:+.4f}")
                
                # Update metrics if calibration improved them
                if paper_results['balanced_accuracy_change'] > 0:
                    log_success(f"🎯 Calibration improved balanced accuracy by {paper_results['balanced_accuracy_change']:.4f}!")
                    acc = paper_results['corrected_accuracy']
                    balanced_acc = paper_results['balanced_accuracy']
                else:
                    log_info("📊 Calibration did not improve balanced accuracy", Colors.CYAN)
            
            # Save detailed calibration results
            calib_path = os.path.join(args.save_dir, f'{ckp}_calibration_analysis.json')
            with open(calib_path, 'w') as f:
                json.dump(convert_numpy_types(calibration_results), f, indent=2)
            log_success(f"Detailed calibration results saved to {calib_path}")
            
            # Log to wandb if enabled
            if args.wandb:
                wandb_metrics = {
                    f"eval_calibration/original_accuracy": acc,
                    f"eval_calibration/original_balanced_accuracy": balanced_acc,
                    f"eval_calibration/checkpoint": ckp
                }
                
                if calib_results.get('paper_calibration'):
                    pc = calib_results['paper_calibration']
                    wandb_metrics.update({
                        f"eval_calibration/paper_accuracy": pc['corrected_accuracy'],
                        f"eval_calibration/paper_balanced_accuracy": pc['balanced_accuracy'],
                        f"eval_calibration/paper_accuracy_improvement": pc['accuracy_improvement'],
                        f"eval_calibration/paper_balanced_accuracy_improvement": pc['balanced_accuracy_change'],
                        f"eval_calibration/gamma": pc['gamma'],
                        f"eval_calibration/absent_accuracy_improvement": pc['absent_accuracy_improvement'],
                        f"eval_calibration/absent_classes_count": len(pc['absent_classes'])
                    })
                
                # Add AUC curve metrics if available
                if auc_curve_data:
                    wandb_metrics.update({
                        f"eval_calibration/auc_seen_improvement": auc_curve_data['seen_improvement'],
                        f"eval_calibration/auc_absent_improvement": auc_curve_data['absent_improvement'],
                        f"eval_calibration/auc_original_seen_acc": auc_curve_data['original_seen_acc'],
                        f"eval_calibration/auc_original_absent_acc": auc_curve_data['original_absent_acc'],
                        f"eval_calibration/auc_calibrated_seen_acc": auc_curve_data['calibrated_seen_acc'],
                        f"eval_calibration/auc_calibrated_absent_acc": auc_curve_data['calibrated_absent_acc']
                    })
                    
                    # Log the plot to wandb
                    if os.path.exists(auc_curve_data['plot_path']):
                        wandb.log({f"eval_calibration/auc_curve_{ckp}": wandb.Image(auc_curve_data['plot_path'])}, step=i)
                
                wandb.log(wandb_metrics, step=i)
                log_success("Calibration metrics and AUC curve logged to wandb")
        else:
            if args.calibration:
                log_warning("Calibration requested but logits not available - skipping calibration analysis")
            calibration_results = {'skipped': True, 'reason': 'Calibration not requested or logits not available'}
        
        # Log metrics with colors
        log_metric("Accuracy", acc, ".4f", Colors.BRIGHT_GREEN)
        log_metric("Balanced Accuracy", balanced_acc, ".4f", Colors.BRIGHT_GREEN)
        log_metric("Evaluation Loss", eval_loss, ".4f", Colors.BRIGHT_BLUE)
        
        # Store results (convert numpy types to Python types for JSON serialization)
        eval_results[ckp] = {
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'loss': float(eval_loss),
            'num_samples': int(len(ckp_eval_dset)),
            'calibration': calibration_results
        }
        
        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                "test/accuracy": acc,
                "test/balanced_accuracy": balanced_acc,
                "test/loss": eval_loss,
                "checkpoint": ckp
            }, step=i)
        
        # Save predictions if requested
        if args.save_predictions:
            pred_path = os.path.join(args.save_dir, f'{ckp}_preds.pkl')
            with open(pred_path, 'wb') as f:
                pickle.dump((preds_arr, labels_arr), f)
            log_success(f"Predictions saved to {pred_path}")
        
        log_subsection_start(f"✅ Checkpoint {ckp} Complete", Colors.BRIGHT_GREEN)
    
    log_section_start("📊 FINAL EVALUATION SUMMARY", Colors.BRIGHT_YELLOW)
    # Save summary results
    summary_path = os.path.join(args.save_dir, 'eval_only_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy_types(eval_results), f, indent=2)
    log_success(f"Summary saved to {summary_path}")
    
    # Calculate average metrics
    if eval_results:
        avg_accuracy = sum(result['accuracy'] for result in eval_results.values()) / len(eval_results)
        avg_balanced_accuracy = sum(result['balanced_accuracy'] for result in eval_results.values()) / len(eval_results)
        avg_loss = sum(result['loss'] for result in eval_results.values()) / len(eval_results)
        total_samples = sum(result['num_samples'] for result in eval_results.values())
    else:
        avg_accuracy = avg_balanced_accuracy = avg_loss = 0.0
        total_samples = 0
    
    # Log summary with beautiful formatting
    logging.info("=== CHECKPOINT RESULTS SUMMARY ===")
    for ckp, results in eval_results.items():
        base_info = f"{ckp}: acc={results['accuracy']:.4f}, balanced_acc={results['balanced_accuracy']:.4f}, loss={results['loss']:.4f}, samples={results['num_samples']}"
        
        # Add calibration info if available
        if 'calibration' in results and results['calibration'].get('paper_calibration'):
            pc = results['calibration']['paper_calibration']
            calib_info = f" | calib: γ={pc['gamma']:.3f}, acc_Δ={pc['accuracy_improvement']:+.4f}, bal_acc_Δ={pc['balanced_accuracy_improvement']:+.4f}"
            base_info += calib_info
        elif 'calibration' in results and results['calibration'].get('skipped', False):
            base_info += f" | calib: {results['calibration']['reason']}"
        
        logging.info(base_info)
    
    logging.info("=" * 60)
    logging.info(f"🎯 FINAL AVERAGE ACROSS ALL CHECKPOINTS:")
    log_metric("Average Accuracy", avg_accuracy, ".4f", Colors.BRIGHT_GREEN)
    log_metric("Average Balanced Accuracy", avg_balanced_accuracy, ".4f", Colors.BRIGHT_GREEN)
    log_metric("Average Loss", avg_loss, ".4f", Colors.BRIGHT_BLUE)
    logging.info(f"  📊 Total Checkpoints: {len(eval_results)}")
    logging.info(f"  📊 Total Samples: {total_samples}")
    
    # Calculate and display calibration improvements if calibration was used
    if args.calibration:
        calibration_improvements = []
        paper_improvements = []
        absent_improvements = []
        checkpoints_with_calibration = 0
        
        for ckp, results in eval_results.items():
            if ('calibration' in results and 
                results['calibration'].get('paper_calibration') is not None and
                not results['calibration'].get('skipped', False)):
                
                checkpoints_with_calibration += 1
                paper_calib = results['calibration']['paper_calibration']
                
                calibration_improvements.append(paper_calib['balanced_accuracy_improvement'])
                paper_improvements.append(paper_calib['accuracy_improvement'])
                absent_improvements.append(paper_calib['absent_accuracy_improvement'])
        
        if checkpoints_with_calibration > 0:
            avg_bal_acc_improvement = np.mean(calibration_improvements)
            avg_acc_improvement = np.mean(paper_improvements)
            avg_absent_improvement = np.mean(absent_improvements)
            
            logging.info("")
            log_subsection_start("📄 PAPER CALIBRATION IMPACT SUMMARY", Colors.BRIGHT_YELLOW)
            log_metric("Average Accuracy Improvement", avg_acc_improvement, "+.4f", 
                     Colors.BRIGHT_GREEN if avg_acc_improvement >= 0 else Colors.BRIGHT_RED)
            log_metric("Average Balanced Accuracy Improvement", avg_bal_acc_improvement, "+.4f",
                     Colors.BRIGHT_GREEN if avg_bal_acc_improvement >= 0 else Colors.BRIGHT_RED)
            log_metric("Average Absent Class Accuracy Improvement", avg_absent_improvement, "+.4f",
                     Colors.BRIGHT_GREEN if avg_absent_improvement >= 0 else Colors.BRIGHT_RED)
            logging.info(f"  📊 Checkpoints with Calibration: {checkpoints_with_calibration}/{len(eval_results)}")
            
            if avg_bal_acc_improvement > 0:
                log_success(f"🎯 Paper calibration method improved balanced accuracy across checkpoints!")
            else:
                log_info("📊 Paper calibration method had minimal impact on balanced accuracy", Colors.CYAN)
        else:
            log_warning("No calibration improvements recorded (calibration may have been skipped)")
    
    logging.info("=" * 60)
    
    # Finalize wandb if enabled
    if args.wandb:
        wandb.finish()
        log_success("✅ W&B run finished")
    
    # Final completion logs
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    log_final_result("🎉 EVALUATION-ONLY MODE COMPLETED SUCCESSFULLY!")
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    
    # Final summary with colors
    log_info(f"📊 Total checkpoints evaluated: {len(eval_results)}")
    log_info(f"💾 Results saved to: {args.save_dir}")
    print(Colors.BOLD + "=" * 80 + Colors.RESET)
    print()
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
    del data, data_test
    
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
    eval_dset = CkpDataset(common_config["eval_data_config_path"], class_names, is_train=False, label_type=label_type)

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

    for i, ckp in enumerate(ckp_list):
        log_subsection_start(f"📊 Evaluating Checkpoint {ckp} ({i+1}/{len(ckp_list)})", Colors.CYAN)
        
        log_step(1, f"Loading model weights", Colors.BLUE)
        # Load model weights for this checkpoint
        model_path = model_file_mapping[ckp]
        
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
                classifier.load_state_dict(state_dict)
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
            
        loss_arr, preds_arr, labels_arr = eval(
            classifier, 
            eval_loader, 
            args.device, 
            chop_head=common_config['chop_head']
        )
        
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
    # Save summary results
    summary_path = os.path.join(args.save_dir, 'eval_only_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    log_success(f"Summary saved to {summary_path}")
    
    # Calculate average metrics
    if eval_results:
        avg_accuracy = sum(result['accuracy'] for result in eval_results.values()) / len(eval_results)
        avg_balanced_accuracy = sum(result['balanced_accuracy'] for result in eval_results.values()) / len(eval_results)
        avg_loss = sum(result['loss'] for result in eval_results.values()) / len(eval_results)
        total_samples = sum(result['num_samples'] for result in eval_results.values())
    else:
        avg_accuracy = avg_balanced_accuracy = avg_loss = 0.0
        total_samples = 0
    
    # Log summary with beautiful formatting
    logging.info("=== CHECKPOINT RESULTS SUMMARY ===")
    for ckp, results in eval_results.items():
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
    
    # Generate AUC curve summary if calibration was used
    if args.calibration:
        log_section_start("📊 CALIBRATION AUC CURVE SUMMARY", Colors.BRIGHT_MAGENTA)
        
        auc_curves_generated = 0
        calibration_improvements = []
        
        for ckp, results in eval_results.items():
            if 'calibration' in results and results['calibration'].get('auc_curve'):
                auc_data = results['calibration']['auc_curve']
                auc_curves_generated += 1
                
                absent_improvement = auc_data['absent_improvement']
                calibration_improvements.append(absent_improvement)
                
                log_info(f"📈 {ckp}: AUC curve generated", Colors.GREEN)
                log_info(f"   Plot: {os.path.basename(auc_data['plot_path'])}", Colors.CYAN)
                log_info(f"   Absent class improvement: {absent_improvement:+.2f}%", 
                        Colors.GREEN if absent_improvement > 0 else Colors.RED)
        
        if auc_curves_generated > 0:
            avg_absent_improvement = sum(calibration_improvements) / len(calibration_improvements)
            
            curve_summary = f"AUC curves generated: {auc_curves_generated}/{len(eval_results)}\n"
            curve_summary += f"Average absent class improvement: {avg_absent_improvement:+.2f}%\n"
            curve_summary += f"Best absent class improvement: {max(calibration_improvements):+.2f}%\n"
            curve_summary += f"Worst absent class improvement: {min(calibration_improvements):+.2f}%"
            
            logging.info(create_info_box("AUC Curve Summary", curve_summary))
            log_info(f"📁 All AUC curves saved to: {args.save_dir}", Colors.CYAN)
        else:
            log_warning("No AUC curves were generated (possibly due to missing seen/absent classes)")

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
    parser.add_argument('--model_dir', type=str, help='Directory containing trained model checkpoints (.pth files) for eval_only mode')
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
                        choices=['bioclip', 'openai'],
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

