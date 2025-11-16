import logging
import wandb  # Ensure wandb is imported
import copy

import os
from contextlib import suppress
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import json

from .loss import CB_loss, focal_loss, standard_focal_loss, LDAM_loss, loss_fn_kd, SupConLoss, cdt_loss, balanced_softmax_loss
from utils.vram_check import check_vram_and_clean, maintenance_vram_check, conditional_cache_clear

# Import epoch logging functions
try:
    from utils.log_formatter import log_epoch_train, log_epoch_val, log_epoch_test, log_training_header, log_training_summary, create_epoch_table_header, log_epoch_table_row
except ImportError:
    # Fallback if log_formatter is not available
    def log_epoch_train(*args, **kwargs): pass
    def log_epoch_val(*args, **kwargs): pass
    def log_epoch_test(*args, **kwargs): pass
    def log_training_header(*args, **kwargs): pass
    def log_training_summary(*args, **kwargs): pass
    def create_epoch_table_header(*args, **kwargs): pass
    def log_epoch_table_row(*args, **kwargs): pass
    def log_training_header(*args, **kwargs): pass
    def log_training_summary(*args, **kwargs): pass
    def create_epoch_table_header(*args, **kwargs): return ""
    def log_epoch_table_row(*args, **kwargs): pass

"""

common.py

    Common utilities for training and evaluation.

"""

def print_metrics(loss_arr, preds_arr, labels_arr, n_classes, log_predix=''):
    # Calculate eval accuracy and loss
    acc = (preds_arr == labels_arr).mean()
    loss = loss_arr.mean()
    # Calculate per-class accuracy
    acc_per_class = []
    for i in range(n_classes):
        mask = labels_arr == i
        if mask.sum() == 0:
            continue
        acc_per_class.append((preds_arr[mask] == labels_arr[mask]).mean())
    acc_per_class = np.array(acc_per_class)
    balanced_acc = acc_per_class.mean()
    logging.info(f'{log_predix}Number of samples: {len(labels_arr)}, acc: {acc:.4f}, balanced acc: {balanced_acc:.4f}, loss: {loss:.4f}. ')
    return acc, balanced_acc

def compute_metrics(loss_arr, preds_arr, labels_arr, n_classes):
    """Compute metrics without logging."""
    # Calculate eval accuracy and loss
    acc = (preds_arr == labels_arr).mean()
    loss = loss_arr.mean()
    # Calculate per-class accuracy
    acc_per_class = []
    for i in range(n_classes):
        mask = labels_arr == i
        if mask.sum() == 0:
            continue
        acc_per_class.append((preds_arr[mask] == labels_arr[mask]).mean())
    acc_per_class = np.array(acc_per_class)
    balanced_acc = acc_per_class.mean()
    return acc, balanced_acc, loss

def compute_IRD(features, temperature, device):
    features_sim = torch.div(torch.matmul(features, features.T), temperature)
    logits_mask = torch.scatter(
            torch.ones_like(features_sim),
            1,
            torch.arange(features_sim.size(0)).view(-1, 1).to(device),
            0
    )
    logits_max, _ = torch.max(features_sim * logits_mask, dim=1, keepdim=True)
    features_sim = features_sim - logits_max.detach()
    row_size = features_sim.size(0)
    logits = torch.exp(
            features_sim[logits_mask.bool()].view(row_size, -1)
    ) / torch.exp(
            features_sim[logits_mask.bool()].view(row_size, -1)
    ).sum(dim=1, keepdim=True)
    return logits

def get_f_loss(loss_type, samples, n_classes, device, alpha=None, beta=None, gamma=None, ref_model=None):
    samples_per_cls = np.zeros(n_classes)
    for sample in samples:
        cls = sample.label
        samples_per_cls[cls] += 1
    samples_per_cls[samples_per_cls == 0] = 1
    if loss_type == 'ce':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            return F.cross_entropy(logits, labels)
        
    elif loss_type == 'focal':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            # Standard focal loss with default parameters
            focal_alpha = alpha if alpha is not None else 1.0  # No class weighting by default
            focal_gamma = gamma if gamma is not None else 2.0  # Standard focusing parameter
            loss = standard_focal_loss(logits, labels, focal_alpha, focal_gamma)
            return loss
    elif loss_type == 'cb-ce':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            cb_beta = beta if beta is not None else 0.9999  # Default beta value
            use_per_class = alpha if alpha is not None else False  # Use alpha to control per-class beta
            # For CB-CE, gamma is not needed, set to 0 or None based on CB_loss implementation
            loss = CB_loss(logits, labels, samples_per_cls, n_classes, 'softmax', cb_beta, 0.0, device, use_per_class_beta=use_per_class)
            return loss
    elif loss_type == 'cb-focal':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            # CB-focal loss with configurable beta strategy
            cb_beta = beta if beta is not None else 0.9999  # Default beta value for global strategy
            cb_gamma = gamma if gamma is not None else 2.0   # Standard focusing parameter
            use_per_class = alpha if alpha is not None else False  # Use alpha to control per-class beta
            loss = CB_loss(logits, labels, samples_per_cls, n_classes, 'focal', cb_beta, cb_gamma, device, use_per_class_beta=use_per_class)
            return loss
    elif loss_type == 'kd':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            alpha = 1.0
            assert ref_model is not None, "Reference model must be provided for KD loss."
            with torch.no_grad():
                ref_outputs = ref_model(images)
            loss  = F.cross_entropy(logits, labels)
            loss += loss_fn_kd(logits, ref_outputs)
            return loss
    elif loss_type == 'supcon':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            bsz = labels.shape[0]
            f1, f2 = torch.split(proj_features, [bsz, bsz], dim=0)
            ff = torch.cat(
                    [f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            train_loss = SupConLoss(
                    ff, device, labels, target_labels=list(range(n_classes)))
            with torch.no_grad():
                ref_features = F.normalize(ref_model.proj_features(images), dim=1)

                logits2 = compute_IRD(ref_features, 0.01, device)
                logits1 = compute_IRD(proj_features,     0.2 , device)
                train_loss += (-logits2 * torch.log(logits1)).sum(1).mean()
            return train_loss
    elif loss_type == 'derpp':
        def f_loss(outputs, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            alpha = 1.0
            beta  = 1.0
            is_buf = is_buf.bool()
            is_new = ~is_buf
            loss  = F.cross_entropy(outputs[is_new], labels[is_new])
            # buffer logits may be empty for new-data rows → mask first dim
            if is_buf.any():
                buf_logits = outputs[is_buf]
                ref_logits = old_logits[is_buf].to(buf_logits.device)
                loss += alpha * F.mse_loss(
                            buf_logits[:, :old_logits.shape[1]],
                            ref_logits.float())
                loss += beta  * F.cross_entropy(buf_logits, labels[is_buf])
            return loss
    elif loss_type == 'cdt':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            cdt_gamma = gamma if gamma is not None else 0.3
            loss = cdt_loss(logits, labels, samples_per_cls, cdt_gamma, device)
            return loss
    elif loss_type == 'ldam':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            ldam_C = alpha if alpha is not None else 0.5  # Use alpha parameter for C
            loss = LDAM_loss(logits, labels, samples_per_cls, n_classes, ldam_C, device)
            return loss
    
    elif loss_type == 'bsm':
        def f_loss(logits, labels, images=None, proj_features=None, old_logits=None, is_buf=None):
            loss = balanced_softmax_loss(logits, labels, samples_per_cls, n_classes, device)
            return loss
    else:
        raise ValueError(f'Unknown loss type {loss_type}. ')
    return f_loss

def get_optimizer(model, optimizer_name, optimizer_params):
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f'Unknown optimizer {optimizer_name}. ')
    return optimizer

def get_scheduler(optimizer, scheduler_name, scheduler_params):
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f'Unknown scheduler {scheduler_name}. ')
    return scheduler

def train(classifier, optimizer, loader, epochs, device, f_loss, eval_per_epoch=False, eval_loader=None, scheduler=None, loss_type=None, train_head_only=False, gpu_monitor=None, save_best_model=True, save_dir=None, model_name_prefix="model", validation_mode="balanced_acc", early_stop_epoch=5, test_per_epoch=False, next_test_loader=None, test_log_prefix='avg_ub_all', test_type="NEXT"):
    # Initialize best model tracking
    best_val_metric = float('-inf')  # For accuracy tracking (higher is better)
    best_val_loss = float('inf')     # For loss (lower is better)
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0  # For early stopping
    
    # Fixed early stopping warmup period (15 epochs)
    early_stop_warmup = 15
    
    # Determine validation mode
    use_loss_for_best = (validation_mode == "loss")
    
    logging.info(f'Training for up to {epochs} epochs')
    if use_loss_for_best:
        logging.info(f'Using validation loss as primary metric for best model selection (lower is better)')
    else:
        logging.info(f'Using validation balanced accuracy as primary metric for best model selection (higher is better)')
    
    if eval_per_epoch and early_stop_epoch > 0:
        logging.info(f'Early stopping warmup: first {early_stop_warmup} epochs will run without early stopping')
        logging.info(f'Early stopping monitoring: after epoch {early_stop_warmup}, will stop if no improvement for {early_stop_epoch} epochs')
    
    # Add training header for better visualization
    log_training_header(model_name_prefix, epochs)
    
    if save_best_model:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f'Best model will be saved to {save_dir} with prefix "{model_name_prefix}"')
        else:
            logging.warning('save_best_model=True but save_dir is None. Model will be kept in memory only.')
    
    # Initialize test logging structures to avoid unbound local errors
    test_results_json = None
    test_log_path = None

    if test_per_epoch and next_test_loader is not None and save_dir is not None:
        # Open test log file for appending
        test_log_path = os.path.join(save_dir, f'{test_log_prefix}_log.json')
        test_results_json = {}

    for epoch in range(epochs):
        # Log memory at epoch start
        if gpu_monitor:
            gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_start")
        
        if test_results_json is not None:
            test_results_json.setdefault(f"epoch_{epoch}", {})
        
        # Additional VRAM debug log for hanging detection
        check_vram_and_clean(context="epoch start")
            
        if train_head_only:
            classifier.visual_model.eval()
            classifier.head.train()
        else:
            classifier.train()
        loss_arr = []
        correct_arr = []
        preds_arr = []
        labels_arr = []
        
        # Set up maintenance VRAM checker
        maintenance_check = maintenance_vram_check(batch_interval=25, threshold=30.0)

        for batch_idx, (inputs, labels, file_paths, old_logits, is_buf) in enumerate(loader):
            # Silent optimized cache clearing and critical warnings only
            if batch_idx % 5 == 0 or batch_idx < 3:
                check_vram_and_clean(context=f"batch {batch_idx}")
                    
                # Silent optimized cache clearing - keeps memory fresh without debug logs
                if batch_idx % 15 == 0:  # Every 15 batches
                    conditional_cache_clear()
            
            # Enhanced VRAM debugging for accumulative training - log every batch in first few epochs
            debug_memory = gpu_monitor and (epoch < 3 or batch_idx < 5)
            
            if debug_memory:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_start", {
                    'batch_size': len(labels) if hasattr(labels, '__len__') else 1,
                    'has_old_logits': old_logits is not None,
                    'has_buffer_samples': is_buf.any() if hasattr(is_buf, 'any') else False
                })
                
            if isinstance(inputs, list):  # for TwoCropTransform
                inputs = torch.cat(inputs, dim=0)
                
            # Move to device and log memory impact
            inputs, labels = inputs.to(device), labels.to(device)
            if debug_memory:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_data_to_device")
            
            # Forward pass with detailed memory tracking
            bsz = labels.size(0)
            if inputs.size(0) == 2 * bsz:
                i1, i2 = torch.split(inputs, [bsz, bsz], dim=0)
                if debug_memory:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_before_forward")
                logits = classifier(i1)
                if debug_memory:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_after_forward")
            else:
                if debug_memory:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_before_forward")
                logits = classifier(inputs)
                if debug_memory:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_after_forward")

            preds = logits.argmax(dim=1)
            correct = preds == labels
            proj_features = None
            if hasattr(classifier, 'proj_head'):
                # For SupCon loss, we need to get the features
                proj_features = classifier.proj_features(inputs)
                
            if debug_memory:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_before_loss")
            loss = f_loss(logits, labels, images=inputs, proj_features=proj_features, old_logits=old_logits, is_buf=is_buf)
            if debug_memory:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_after_loss", {
                    'loss_value': loss.item()
                })
            
            # Backward pass with memory tracking
            optimizer.zero_grad()
            if debug_memory:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_before_backward")
            loss.backward()
            # gradient tracking
            if debug_memory:
                for name, param in classifier.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        logging.info(f"Gradient flow check: Param: {name} -- {param.shape}")
            if debug_memory:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_after_backward")
            optimizer.step()
            if debug_memory:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_after_optimizer_step")
            
            # Append arrays before cleanup (need to access tensors)
            loss_arr.append(loss.cpu().item())
            correct_arr.append(correct.cpu().numpy())
            preds_arr.append(preds.cpu().numpy())
            labels_arr.append(labels.cpu().numpy())
            
            # Clean up tensors to help with memory
            del inputs, labels, logits, loss
            if proj_features is not None:
                del proj_features
            
            # Safe cache clearing after batch completion - only when really needed
            if debug_memory:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_after_cleanup")
            else:
                # Every 25 batches, silent maintenance cleaning
                maintenance_check(batch_idx)
        
        # Debug log after completing all batches
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            
            # Safe epoch-end cache clearing if memory is high
            conditional_cache_clear(threshold=22.0)
        
        loss_arr = np.array(loss_arr)
        correct_arr = np.concatenate(correct_arr, axis=0)
        preds_arr = np.concatenate(preds_arr, axis=0)
        labels_arr = np.concatenate(labels_arr, axis=0)
        
        # Calculate overall accuracy and balanced accuracy
        avg_acc = correct_arr.mean()
        avg_loss = loss_arr.mean()
        
        # Calculate balanced accuracy using the same logic as print_metrics
        n_classes = len(loader.dataset.class_names) if hasattr(loader.dataset, 'class_names') else len(set(labels_arr))
        acc_per_class = []
        for i in range(n_classes):
            mask = labels_arr == i
            if mask.sum() == 0:
                continue
            acc_per_class.append((preds_arr[mask] == labels_arr[mask]).mean())
        
        if len(acc_per_class) > 0:
            balanced_acc = np.array(acc_per_class).mean()
        else:
            balanced_acc = avg_acc  # Fallback if no classes found
        
        # Use new epoch logging format
        log_epoch_train(epoch, avg_loss, avg_acc, balanced_acc, optimizer.param_groups[0]["lr"])

        if test_results_json is not None:
            test_results_json[f"epoch_{epoch}"]['train'] = {
                'loss': float(avg_loss),
                'acc': float(avg_acc),
                'balanced_acc': float(balanced_acc)
            }

        # Log memory at epoch end
        if gpu_monitor:
            gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_end", {
                'epoch_loss': avg_loss,
                'epoch_acc': avg_acc
            })

        # Log training loss and accuracy to wandb if initialized
        try:
            import wandb
            if wandb.run is not None:  # Check if wandb is initialized
                wandb.log({
                    "train/loss": avg_loss,
                    "train/acc": avg_acc,
                    "train/balanced_acc": balanced_acc,
                    "train/learning_rate": optimizer.param_groups[0]["lr"]
                })
        except (ImportError, AttributeError):
            pass  # wandb not available or not initialized

        if scheduler is not None:
            scheduler.step()

        if eval_per_epoch:
            # Log memory before evaluation
            if gpu_monitor:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_before_eval")

            loss_arr, preds_arr, labels_arr, _, _ = eval(classifier, eval_loader, device)

            # Log memory after evaluation
            if gpu_monitor:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_after_eval", {
                    'eval_samples': len(labels_arr)
                })
                
            # Compute metrics without logging to avoid duplication
            eval_acc, eval_balanced_acc, eval_loss = compute_metrics(loss_arr, preds_arr, labels_arr, len(eval_loader.dataset.class_names))
            
            if test_results_json is not None:
                test_results_json[f"epoch_{epoch}"]['val'] = {
                    'loss': float(eval_loss),
                    'acc': float(eval_acc),
                    'balanced_acc': float(eval_balanced_acc)
                }

            # Check if this is the best model so far
            current_val_loss = eval_loss
            current_val_balanced_acc = eval_balanced_acc
            
            is_best = False
            improved = False
            best_indicator = ""
            
            if use_loss_for_best:
                # Use validation loss as primary metric (lower is better)
                if current_val_loss < best_val_loss:
                    is_best = True
                    improved = True
                    best_val_loss = current_val_loss
                    best_val_metric = current_val_balanced_acc  # Store balanced acc for logging
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_indicator = f"BEST LOSS (↓) SAVED"
                else:
                    # Only increment epochs_without_improvement after warmup period
                    if epoch >= early_stop_warmup:
                        epochs_without_improvement += 1
            else:
                # Use validation balanced accuracy as primary metric (higher is better)
                # For same balanced accuracy, use the earliest one (strict >)
                if current_val_balanced_acc > best_val_metric:
                    is_best = True
                    improved = True
                    best_val_metric = current_val_balanced_acc
                    best_val_loss = current_val_loss  # Store loss for logging
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_indicator = f"BEST ACC (↑) SAVED"
                else:
                    # Only increment epochs_without_improvement after warmup period
                    if epoch >= early_stop_warmup:
                        epochs_without_improvement += 1
            
            # Use new validation logging format with best model indicator
            log_epoch_val(epoch, eval_loss, eval_acc, eval_balanced_acc, len(labels_arr), is_best, best_indicator)
            
            # Test evaluation per epoch if enabled
            if test_per_epoch and next_test_loader is not None:
                # Log memory before test evaluation
                if gpu_monitor:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_before_test_eval")
                    
                if isinstance(next_test_loader, dict):
                    # Upper bound mode: evaluate on each test checkpoint separately and compute average
                    test_results = []
                    total_samples = 0
                    
                    for ckp_name, test_loader in next_test_loader.items():
                        ckp_test_loss_arr, ckp_test_preds_arr, ckp_test_labels_arr, _, _ = eval(classifier, test_loader, device)
                        ckp_test_acc, ckp_test_balanced_acc, ckp_test_loss = compute_metrics(
                            ckp_test_loss_arr, ckp_test_preds_arr, ckp_test_labels_arr, 
                            len(test_loader.dataset.class_names)
                        )
                        test_results.append({
                            'checkpoint': ckp_name,
                            'acc': ckp_test_acc,
                            'balanced_acc': ckp_test_balanced_acc,
                            'loss': ckp_test_loss,
                            'samples': len(ckp_test_labels_arr)
                        })
                        test_results_json[f"epoch_{epoch}"]["test_"+ckp_name] = {
                            'acc': float(ckp_test_acc),
                            'balanced_acc': float(ckp_test_balanced_acc),
                            'loss': float(ckp_test_loss),
                        }
                        total_samples += len(ckp_test_labels_arr)
                    
                    # Compute averages across all test checkpoints
                    if test_results:
                        avg_test_acc = np.mean([r['acc'] for r in test_results])
                        avg_test_balanced_acc = np.mean([r['balanced_acc'] for r in test_results])
                        avg_test_loss = np.mean([r['loss'] for r in test_results])
                        test_results_json[f"epoch_{epoch}"]['test_avg'] = {
                            'avg_acc': float(avg_test_acc),
                            'avg_balanced_acc': float(avg_test_balanced_acc),
                            'avg_loss': float(avg_test_loss),
                        }
                        
                        # Log the averaged results
                        log_epoch_test(epoch, avg_test_loss, avg_test_acc, avg_test_balanced_acc, total_samples, "UB_AVG")
                else:
                    # Single test loader mode (accumulative or other modes)
                    next_test_loss_arr, next_test_preds_arr, next_test_labels_arr, _, _ = eval(classifier, next_test_loader, device)
                    next_test_acc, next_test_balanced_acc, next_test_loss = compute_metrics(
                        next_test_loss_arr, next_test_preds_arr, next_test_labels_arr, 
                        len(next_test_loader.dataset.class_names)
                    )
                    test_results_json[f"epoch_{epoch}"]['test'] = {
                        'acc': float(next_test_acc),
                        'balanced_acc': float(next_test_balanced_acc),
                        'loss': float(next_test_loss),
                    }
                    log_epoch_test(epoch, next_test_loss, next_test_acc, next_test_balanced_acc, len(next_test_labels_arr), test_type)

                # Log memory after test evaluation
                if gpu_monitor:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_after_test_eval")
            
            # Add separator line after each epoch's complete log set
            if eval_per_epoch or test_per_epoch:
                logging.info("─" * 80)
            
            # Save best model state (without logging - already shown in epoch log)
            if is_best and save_best_model:
                # Log memory before model state saving
                if gpu_monitor:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_before_save_best_model")
                    
                best_model_state = copy.deepcopy(classifier.state_dict())
                
                # Log memory after model state copying
                if gpu_monitor:
                    gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_after_copy_best_model")
                    
                if save_dir:
                    best_model_path = os.path.join(save_dir, f'{model_name_prefix}_best_model.pth')
                    torch.save(best_model_state, best_model_path)
                    # If saving to disk, we can clean up immediately and load later from disk
                    del best_model_state
                    best_model_state = None  # Mark as saved to disk
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Log memory after model state cleanup
                    if gpu_monitor:
                        gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_after_cleanup_best_model")
            
            # Early stopping check (only after warmup period)
            if early_stop_epoch > 0 and epoch >= early_stop_warmup and epochs_without_improvement >= early_stop_epoch:
                logging.info(f'Early stopping triggered: no improvement for {early_stop_epoch} epochs after warmup period')
                logging.info(f'Warmup period completed: first {early_stop_warmup} epochs, monitoring started from epoch {early_stop_warmup}')
                logging.info(f'Early stopping triggered at epoch {epoch} (after {epochs_without_improvement} epochs without improvement)')
                logging.info(f'Latest model would have stopped at epoch {early_stop_warmup + early_stop_epoch} if no improvement')
                if use_loss_for_best:
                    logging.info(f'Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}')
                else:
                    logging.info(f'Best epoch was {best_epoch} with val_balanced_acc={best_val_metric:.4f}')
                break
            
            # Log evaluation metrics to wandb if initialized
            try:
                import wandb
                if wandb.run is not None:
                    eval_loss = loss_arr.mean()
                    if epoch < early_stop_warmup:
                        status_indicator = "(WARMUP)" if not is_best else "(BEST-WARMUP)"
                    else:
                        status_indicator = "(BEST)" if is_best else f"({epochs_without_improvement}/{early_stop_epoch})" if early_stop_epoch > 0 else ""
                    logging.info(f'Epoch {epoch}: train_acc={avg_acc:.4f}, train_balanced_acc={balanced_acc:.4f}, val_acc={eval_acc:.4f}, val_balanced_acc={eval_balanced_acc:.4f}, val_loss={eval_loss:.4f}, LR={optimizer.param_groups[0]["lr"]:.8f} {status_indicator}')
                    wandb.log({
                        "val/loss": eval_loss,
                        "val/acc": eval_acc,
                        "val/balanced_acc": eval_balanced_acc,
                        "val/is_best": is_best,
                        "val/best_epoch": best_epoch,
                        "val/epochs_without_improvement": epochs_without_improvement,
                        "val/validation_mode": validation_mode,
                        "val/in_warmup": epoch < early_stop_warmup,
                        "val/warmup_epoch": early_stop_warmup
                    })
            except (ImportError, AttributeError):
                pass  # wandb not available or not initialized
        else:
            # If not evaluating per epoch, just continue training
            pass
    
    if test_per_epoch and next_test_loader is not None and save_dir is not None:
        logging.info(f'Test results logged to {test_log_path}')
        with open(test_log_path, 'w') as f:
            json.dump(test_results_json, f, indent=2)

    # Load best model if we saved one
    if save_best_model and (best_model_state is not None or save_dir):
        if best_model_state is not None:
            # Load from memory
            classifier.load_state_dict(best_model_state)
            # Clean up best model state from memory
            del best_model_state
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif save_dir:
            # Load from disk
            best_model_path = os.path.join(save_dir, f'{model_name_prefix}_best_model.pth')
            if os.path.exists(best_model_path):
                state_dict = torch.load(best_model_path, map_location=device)
                classifier.load_state_dict(state_dict)
                del state_dict  # Clean up loaded state
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if use_loss_for_best:
            logging.info(f'Loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f}, val_balanced_acc={best_val_metric:.4f})')
        else:
            logging.info(f'Loaded best model from epoch {best_epoch} (val_balanced_acc={best_val_metric:.4f}, val_loss={best_val_loss:.4f})')
        
        # Log final best model info to wandb
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "final/final_best_epoch": best_epoch,
                    "final/final_best_val_loss": best_val_loss,
                    "final/final_best_val_balanced_acc": best_val_metric,
                    "final/total_epochs_trained": epoch + 1,
                    "final/training_completed": True,
                    "final/validation_mode": validation_mode,
                    "final/early_stopped": epochs_without_improvement >= early_stop_epoch if early_stop_epoch > 0 else False
                })
        except (ImportError, AttributeError):
            pass
    
    # Add training completion summary
    best_metric_name = "val_loss" if use_loss_for_best else "val_balanced_acc"
    best_metric_value = best_val_loss if use_loss_for_best else best_val_metric
    log_training_summary(best_epoch, best_metric_value, best_metric_name)
    
    return classifier

def eval(classifier, loader, device, chop_head=False, return_logits=False):
    dset = loader.dataset
    if len(dset) == 0:
        if return_logits:
            return np.array([]), np.array([]), np.array([]), np.array([]), [], []
        else:
            return np.array([]), np.array([]), np.array([]), [], []
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

    classifier.eval()
    pred_false = []
    pred_true = []
    with torch.no_grad():
        for inputs, labels, file_paths, _, _ in loader:
            # Forward
            inputs, labels = inputs.to(device), labels.to(device)
            logits = classifier(inputs)
            loss = F.cross_entropy(logits, labels, reduction='none')
            logits[:, chop_mask] = -np.inf
            confidences = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            # Collect false/true predictions for analysis
            for i in range(len(labels)):
                if preds[i] == labels[i]:
                    pred_true.append((file_paths[i], labels[i].item(), preds[i].item(), confidences[i].max().item()))
                else:
                    pred_false.append((file_paths[i], labels[i].item(), preds[i].item(), confidences[i].max().item()))
            # Append
            loss_arr.append(loss.cpu().numpy())
            preds_arr.append(preds.cpu().numpy())
            labels_arr.append(labels.cpu().numpy())
            if return_logits:
                logits_arr.append(logits.cpu().numpy())
    loss_arr = np.concatenate(loss_arr, axis=0)
    preds_arr = np.concatenate(preds_arr, axis=0)
    labels_arr = np.concatenate(labels_arr, axis=0)
    if return_logits:
        logits_arr = np.concatenate(logits_arr, axis=0)
        return loss_arr, preds_arr, labels_arr, logits_arr, pred_true, pred_false
    else:
        return loss_arr, preds_arr, labels_arr, pred_true, pred_false


def load_classifier(classifier, path):
    logging.info(f'Loading classifier from {path}... ')
    classifier.load_state_dict(torch.load(path))


def save_classifier(classifier, path):
    if not os.path.exists(os.path.dirname(path)):
        logging.info(f'Creating directory {os.path.dirname(path)}... ')
        os.makedirs(os.path.dirname(path))
    logging.info(f'Saving classifier to {path}... ')
    torch.save(classifier.state_dict(), path)


def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def load_best_model(classifier, save_dir, model_name_prefix="model", device="cuda"):
    """Load the best model weights from saved file.
    
    Args:
        classifier (nn.Module): The model to load weights into.
        save_dir (str): Directory where the best model was saved.
        model_name_prefix (str): Prefix used when saving the model.
        device (str): Device to load the model on.
    
    Returns:
        bool: True if model was loaded successfully, False otherwise.
    """
    if not save_dir or not os.path.exists(save_dir):
        logging.warning(f'Save directory {save_dir} does not exist. Cannot load best model.')
        return False
    
    best_model_path = os.path.join(save_dir, f'{model_name_prefix}_best_model.pth')
    
    if not os.path.exists(best_model_path):
        logging.warning(f'Best model file {best_model_path} does not exist. Cannot load best model.')
        return False
    
    try:
        state_dict = torch.load(best_model_path, map_location=device)
        classifier.load_state_dict(state_dict)
        logging.info(f'Successfully loaded best model from {best_model_path}')
        return True
    except Exception as e:
        logging.error(f'Failed to load best model from {best_model_path}: {e}')
        return False
