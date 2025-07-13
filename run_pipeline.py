import argparse
import logging
import os
import copy
import time

from datetime import datetime
import numpy as np
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader
import pprint
import pickle
import wandb  # Ensure wandb is imported

from core import *
from core.module import get_al_module, get_cl_module, get_ood_module
from utils.misc import method_name
from utils.gpu_monitor import get_gpu_monitor, log_gpu_memory, monitor_model_memory

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
    # Use deterministic timestamp based on configuration for reproducibility
    deterministic_time = f"2025-07-12-20-{abs(hash(str(vars(params)))) % 10000:04d}-00"
    petl_method_name = method_name(params)
    log_path = os.path.join(log_path, params.pretrained_weights)
    petl_method_name = petl_method_name + f'_text_{params.text}'
    if params.interpolation_model:
        petl_method_name += f'_interpolation_model_{params.interpolation_alpha}'
    log_path = os.path.join(log_path, petl_method_name)
    log_path = os.path.join(log_path, deterministic_time)
    if not debug:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_path, 'log')
    else:
        logger.setLevel(logging.DEBUG)
        curr_time = f"debug-{deterministic_time}"
        log_path = os.path.join(log_path, curr_time)
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
    return log_path

def pretrain(classifier, class_names, pretrain_config, common_config, device, gpu_monitor=None, interpolation_model=False, interpolation_head=False, interpolation_alpha=0.5, eval_per_epoch=False, save_dir=None, args=None):
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
        
        # Log class distribution in validation set
        val_class_counts = defaultdict(int)
        train_class_counts = defaultdict(int)
        for sample in val_samples:
            val_class_counts[sample.label] += 1
        for sample in train_samples:
            train_class_counts[sample.label] += 1
        
        logging.info(f'Pretraining validation split (2 samples per class):')
        logging.info(f'{"Class":<20} {"Train":<8} {"Val":<8} {"Val%":<8}')
        logging.info(f'{"-"*44}')
        for class_idx in sorted(set(list(val_class_counts.keys()) + list(train_class_counts.keys()))):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            train_count = train_class_counts[class_idx]
            val_count = val_class_counts[class_idx]
            total_count = train_count + val_count
            val_percentage = (val_count / total_count * 100) if total_count > 0 else 0
            logging.info(f'{class_name:<20} {train_count:<8} {val_count:<8} {val_percentage:<8.1f}%')
        
        total_train = sum(train_class_counts.values())
        total_val = sum(val_class_counts.values())
        total_all = total_train + total_val
        overall_val_percentage = (total_val / total_all * 100) if total_all > 0 else 0
        
        logging.info(f'{"-"*44}')
        logging.info(f'{"TOTAL":<20} {total_train:<8} {total_val:<8} {overall_val_percentage:<8.1f}%')
        logging.info(f'Validation classes: {len(val_class_counts)} out of {len(class_names)} possible classes')
        
        eval_batch_size = common_config.get('eval_batch_size', train_batch_size)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
        logging.info(f'Pretrain eval dataset size: {len(eval_dataset)} (2 samples per class from full training data). ')
        logging.info(f'Updated pretrain training dataset size: {len(dataset)} (excluding validation samples). ')
    
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
    
    # Train
    train(classifier, optimizer, loader, epochs, device, f_loss, 
          scheduler=scheduler, 
          gpu_monitor=gpu_monitor, 
          eval_per_epoch=eval_per_epoch, 
          eval_loader=eval_loader,
          save_best_model=save_best_model,
          save_dir=save_dir,
          model_name_prefix="pretrain")
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
    # Initialize GPU memory monitoring if enabled
    gpu_monitor = None
    if args.gpu_memory_monitor:
        enable_colors = not args.no_gpu_monitor_colors  # Colors enabled by default, disabled with --no_gpu_monitor_colors
        gpu_monitor = get_gpu_monitor(args.device, args.wandb, enable_colors)
        logging.info("GPU memory monitoring enabled.")
        gpu_monitor.log_memory_usage("startup", "initial")
    
    # Initialize wandb if enabled
    if args.wandb:
        import re
        
        # Extract components from the original save_dir
        match = re.search(r"pipeline/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)", args.save_dir)
        wandb_run_name = "Unidentified Run"  # Default name if regex fails
        if match:
            dataset = match.group(1)  # e.g., ENO_C05_new
            training_mode = match.group(2)  # e.g., ce
            pretrained_type = match.group(3)  # e.g., accumulative-scratch
            pretrained_weights = match.group(4)  # e.g., bioclip2
            method = match.group(5)  # e.g., lora_8_text_lora
            timestamp = match.group(6)  # e.g., 2025-07-05-14-53-19

            # Construct the wandb run name
            wandb_run_name = f"{dataset} | {training_mode} | {pretrained_type} | {pretrained_weights} | {method} | {timestamp}"

        module_name = getattr(args, 'module_name', 'default_module')  # Fallback if module_name is not in args
        wandb.init(
            project="ICICLE-Benchmark-LR-FINAL-SANITY-LOSS-BEST",  # Replace with your project name
            name=wandb_run_name,  # Set run name using args.c and module_name
            config=vars(args)  # Log all arguments to wandb
        )
        logging.info("wandb logging is enabled.")

    # Print args
    logging.info(pprint.pformat(vars(args)))
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
    
    # Load model
    classifier = build_classifier(args, class_names, args.device)
    
    # Monitor model memory usage if enabled
    if args.gpu_memory_monitor:
        gpu_monitor.log_memory_usage("model_load", "after_build")
        monitor_model_memory(classifier, "classifier", args.device, args.wandb)
        gpu_monitor.clear_cache_and_log("model_load")
    
    # Pretrain
    if pretrain_config['pretrain']:
        logging.info('Pretraining classifier... ')
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
                              args=args)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("pretrain", "after")
            gpu_monitor.clear_cache_and_log("pretrain")
    else:
        logging.info('Skipping pretraining... ')
    
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
    logging.info(f'Checkpoint list, length: {len(ckp_list)}: ')
    for ckp in ckp_list:
        logging.info(f'\t{ckp} ')
    
    # Initialize modules
    ood_module = get_ood_module(ood_config, common_config, class_names, args, args.device)
    al_module = get_al_module(al_config, common_config, class_names, args, args.device)
    cl_module = get_cl_module(classifier, cl_config, common_config, class_names, args, args.device)
    
    # Main loop
    for i in range(len(ckp_list)):
        # Get checkpoint
        ckp_prev = ckp_list[i - 1] if i > 0 else None
        ckp = ckp_list[i]
        logging.info(f'Training on checkpoint {ckp}. ')
        
        # Monitor memory at checkpoint start
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("checkpoint", f"start_{ckp}")
        
        # Get training and evaluation dataset
        ckp_train_dset = train_dset.get_subset(is_train=True, ckp_list=ckp_prev)
        ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp)
        logging.info(f'Training dataset size: {len(ckp_train_dset)}. ')
        logging.info(f'Evaluation dataset size: {len(ckp_eval_dset)}. ')
        
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
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("ood", f"before_{ckp}")
        classifier, ood_mask = ood_module.process(classifier, ckp_train_dset, ckp_eval_dset, train_dset_mask)
        if args.gpu_memory_monitor:
            gpu_monitor.log_memory_usage("ood", f"after_{ckp}")
        logging.info(f'OOD mask: {ood_mask.sum()} / {len(ood_mask)}. ')
        
        # Run active learning
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
        logging.info(f'AL mask: {al_mask.sum()} / {len(al_mask)}. ')

        # Prepare evaluation dataloader
        cl_eval_loader = DataLoader(ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False, worker_init_fn=worker_init_fn)

        # Prepare validation loader for continual learning
        cl_validation_loader = None
        if args.eval_per_epoch:
            # Check if we're using accumulative training (which trains incrementally)
            cl_method = cl_config.get('method', 'none')
            if 'accumulative' in cl_method:
                # For accumulative training: use current checkpoint's test data as validation
                cl_validation_loader = cl_eval_loader
                logging.info(f'Using current checkpoint {ckp} test data as validation for accumulative training')
            else:
                # For other methods: keep using the regular evaluation loader
                cl_validation_loader = cl_eval_loader
                logging.info(f'Using regular evaluation data as validation for {cl_method}')

        # Run continual learning
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

        if not pretrain_config['pretrain'] and (args.interpolation_model or args.interpolation_head):
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("interpolation", f"before_{ckp}")
            # Interpolate model if enabled
            if args.interpolation_model:
                logging.info(f'Interpolating model at checkpoint {ckp} with alpha {args.interpolation_alpha}. ')
                classifier.interpolate_model(cl_module._classifier, alpha=args.interpolation_alpha)
            if args.interpolation_head:
                logging.info(f'Interpolating head at checkpoint {ckp} with alpha {args.interpolation_alpha}. ')
                classifier.interpolate_head(cl_module._classifier, alpha=args.interpolation_alpha)

        # Save model and predictions
        acc, balanced_acc = 0.0, 0.0  # Initialize metrics
        eval_loss = 0.0  # Initialize eval_loss
        
        if args.accu_eval:
            logging.info(f'Accu-eval start on {ckp_list[i]}')
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
                a, b = print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"Accu-eval start on {ckp_list[i]} at {eval_ckp}: ")
                
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
                logging.info(f'Evaluating on current checkpoint {ckp_prev}. ')
                current_ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp_prev)
                current_cl_eval_loader = DataLoader(current_ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False)
                loss_arr, preds_arr, labels_arr = eval(classifier, current_cl_eval_loader, args.device, chop_head=common_config['chop_head'])
                print_metrics(loss_arr, preds_arr, labels_arr, len(class_names), log_predix=f"Eval on current training checkpoint {ckp_prev}: ")
            logging.info(f'Evaluating on next checkpoint {ckp}. ')
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"before_{ckp}")
            loss_arr, preds_arr, labels_arr = eval(classifier, cl_eval_loader, args.device, chop_head=common_config['chop_head'])
            if args.gpu_memory_monitor:
                gpu_monitor.log_memory_usage("evaluation", f"after_{ckp}")
            acc, balanced_acc = print_metrics(loss_arr, preds_arr, labels_arr, len(class_names))
            eval_loss = np.mean(loss_arr)

        # Log training and evaluation loss to wandb
        if args.wandb:
            logging.info(f"Logging main metrics to wandb: ckp={ckp}, acc={acc:.4f}, balanced_acc={balanced_acc:.4f}, eval_loss={eval_loss:.4f}")
            wandb.log({
                "test/accuracy": acc,  # Overall accuracy
                "test/balanced_accuracy": balanced_acc,  # Balanced accuracy
                "test/loss": eval_loss,  # Evaluation loss
                "checkpoint": ckp
            }, step=i)

        if args.is_save:
            logging.info(f'Saving model to {args.save_dir}. ')
            save_path = os.path.join(args.save_dir, f'{ckp}.pth')
            torch.save(classifier.state_dict(), save_path)
        logging.info(f"Saving predictions to {args.save_dir}. ")
        pred_path = os.path.join(args.save_dir, f'{ckp}_preds.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump((preds_arr, labels_arr), f)
        mask_path = os.path.join(args.save_dir, f'{ckp}_mask.pkl')
        with open(mask_path, 'wb') as f:
            pickle.dump((ood_mask, al_mask), f)
        
        # Clear GPU cache between checkpoints
        if args.gpu_memory_monitor:
            gpu_monitor.clear_cache_and_log(f"checkpoint_end_{ckp}")

    # Final GPU memory summary
    if args.gpu_memory_monitor:
        summary = gpu_monitor.get_memory_summary()
        logging.info(f"GPU Memory Summary: {summary}")
        if args.wandb:
            wandb.log({"gpu_memory/summary": summary})

    # Finalize wandb if enabled
    if args.wandb:
        wandb.finish()

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
    parser.add_argument('--save_best_model', action='store_true', default=True, help='Save best model during training')
    parser.add_argument('--is_save', action='store_true', help='Save model')
    parser.add_argument('--eval_only', action='store_true', help='Evaluate only')
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
                        choices=['head', 'petl', 'full'],
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

    ########################block#########################
    parser.add_argument('--block_index', default=None, type=int, nargs='+',
                        help='indexes of block to tune (default: %(default)s)')

    ########################domain generalization#########################
    parser.add_argument('--generalization_test', type=str, default='a',
                        choices=['v2', 's', 'a'],
                        help='domain generalization test set for imagenet')
    parser.add_argument('--merge_factor', default=1, type=float,
                        help='merge factor')

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
    args.save_dir = setup_logging(args.log_path, args.debug, args)
    logging.info(f'Saving to {args.save_dir}. ')

    # Save configuration
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yml = yaml.YAML()
        yml.dump(vars(args), f)

    # Run
    start_time = time.time()
    run(args)
    end_time = time.time()

    # Print elapsed time
    logging.info(f'Elapsed time: {end_time - start_time:.2f} seconds. ')

