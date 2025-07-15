import logging
import wandb  # Ensure wandb is imported
import copy

import os
from contextlib import suppress
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .loss import CB_loss, focal_loss, standard_focal_loss, LDAM_loss, loss_fn_kd, SupConLoss, cdt_loss, balanced_softmax_loss

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

def train(classifier, optimizer, loader, epochs, device, f_loss, eval_per_epoch=False, eval_loader=None, scheduler=None, loss_type=None, train_head_only=False, gpu_monitor=None, save_best_model=True, save_dir=None, model_name_prefix="model", validation_mode="balanced_acc", early_stop_epoch=10):
    # Initialize best model tracking
    best_val_metric = float('-inf')  # For accuracy tracking (higher is better)
    best_val_loss = float('inf')     # For loss (lower is better)
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0  # For early stopping
    
    # Determine validation mode
    use_loss_for_best = (validation_mode == "loss")
    
    logging.info(f'Training for up to {epochs} epochs')
    if use_loss_for_best:
        logging.info(f'Using validation loss as primary metric for best model selection (lower is better)')
    else:
        logging.info(f'Using validation balanced accuracy as primary metric for best model selection (higher is better)')
    
    if eval_per_epoch and early_stop_epoch > 0:
        logging.info(f'Early stopping enabled: will stop if no improvement for {early_stop_epoch} epochs')
    
    if save_best_model:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f'Best model will be saved to {save_dir} with prefix "{model_name_prefix}"')
        else:
            logging.warning('save_best_model=True but save_dir is None. Model will be kept in memory only.')
    
    for epoch in range(epochs):
        # Log memory at epoch start
        if gpu_monitor:
            gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_start")
            
        if train_head_only:
            classifier.visual_model.eval()
            classifier.head.train()
        else:
            classifier.train()
        loss_arr = []
        correct_arr = []
        preds_arr = []
        labels_arr = []
        for batch_idx, (inputs, labels, old_logits, is_buf) in enumerate(loader):
            # Log memory for first few batches
            if gpu_monitor and batch_idx < 3:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_before")
                
            if isinstance(inputs, list):  # for TwoCropTransform
                inputs = torch.cat(inputs, dim=0)
            # Forward
            inputs, labels = inputs.to(device), labels.to(device)
            bsz = labels.size(0)
            if inputs.size(0) == 2 * bsz:
                i1, i2 = torch.split(inputs, [bsz, bsz], dim=0)
                logits = classifier(i1)
            else:
                logits = classifier(inputs)
            preds = logits.argmax(dim=1)
            correct = preds == labels
            proj_features = None
            if hasattr(classifier, 'proj_head'):
                # For SupCon loss, we need to get the features
                proj_features = classifier.proj_features(inputs)
            loss = f_loss(logits, labels, images=inputs, proj_features=proj_features, old_logits=old_logits, is_buf=is_buf)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log memory for first few batches
            if gpu_monitor and batch_idx < 3:
                gpu_monitor.log_memory_usage("training", f"epoch_{epoch}_batch_{batch_idx}_after")
            
            # Append
            loss_arr.append(loss.cpu().item())
            correct_arr.append(correct.cpu().numpy())
            preds_arr.append(preds.cpu().numpy())
            labels_arr.append(labels.cpu().numpy())
        
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
        
        logging.info(f'Epoch {epoch}, loss: {avg_loss:.4f}, acc: {avg_acc:.4f}, balanced_acc: {balanced_acc:.4f}, lr: {optimizer.param_groups[0]["lr"]:.8f}. ')

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
            loss_arr, preds_arr, labels_arr = eval(classifier, eval_loader, device)
            eval_acc, eval_balanced_acc = print_metrics(loss_arr, preds_arr, labels_arr, len(eval_loader.dataset.class_names), log_predix=f'Epoch {epoch} val, ')
            
            # Check if this is the best model so far
            current_val_loss = loss_arr.mean()
            current_val_balanced_acc = eval_balanced_acc
            
            is_best = False
            improved = False
            
            if use_loss_for_best:
                # Use validation loss as primary metric (lower is better)
                if current_val_loss < best_val_loss:
                    is_best = True
                    improved = True
                    best_val_loss = current_val_loss
                    best_val_metric = current_val_balanced_acc  # Store balanced acc for logging
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
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
                else:
                    epochs_without_improvement += 1
            
            # Save best model state
            if is_best and save_best_model:
                best_model_state = copy.deepcopy(classifier.state_dict())
                if save_dir:
                    best_model_path = os.path.join(save_dir, f'{model_name_prefix}_best_model.pth')
                    torch.save(best_model_state, best_model_path)
                    if use_loss_for_best:
                        logging.info(f'New best model saved at epoch {epoch}: val_loss={current_val_loss:.4f} (↓), val_balanced_acc={current_val_balanced_acc:.4f} -> {best_model_path}')
                    else:
                        logging.info(f'New best model saved at epoch {epoch}: val_balanced_acc={current_val_balanced_acc:.4f} (↑), val_loss={current_val_loss:.4f} -> {best_model_path}')
            
            # Early stopping check
            if early_stop_epoch > 0 and epochs_without_improvement >= early_stop_epoch:
                logging.info(f'Early stopping triggered: no improvement for {early_stop_epoch} epochs')
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
                    status_indicator = "(BEST)" if is_best else f"({epochs_without_improvement}/{early_stop_epoch})" if early_stop_epoch > 0 else ""
                    logging.info(f'Epoch {epoch}: train_acc={avg_acc:.4f}, train_balanced_acc={balanced_acc:.4f}, val_acc={eval_acc:.4f}, val_balanced_acc={eval_balanced_acc:.4f}, val_loss={eval_loss:.4f}, LR={optimizer.param_groups[0]["lr"]:.8f} {status_indicator}')
                    wandb.log({
                        "val/loss": eval_loss,
                        "val/acc": eval_acc,
                        "val/balanced_acc": eval_balanced_acc,
                        "val/is_best": is_best,
                        "val/best_epoch": best_epoch,
                        "val/epochs_without_improvement": epochs_without_improvement,
                        "val/validation_mode": validation_mode
                    })
            except (ImportError, AttributeError):
                pass  # wandb not available or not initialized
        else:
            # If not evaluating per epoch, just continue training
            pass
    
    # Load best model if we saved one
    if save_best_model and best_model_state is not None:
        classifier.load_state_dict(best_model_state)
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
    
    return classifier

def eval(classifier, loader, device, chop_head=False, return_logits=False):
    dset = loader.dataset
    if len(dset) == 0:
        if return_logits:
            return np.array([]), np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([]), np.array([])
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
    with torch.no_grad():
        for inputs, labels, _, _ in loader:
            # Forward
            inputs, labels = inputs.to(device), labels.to(device)
            logits = classifier(inputs)
            loss = F.cross_entropy(logits, labels, reduction='none')
            logits[:, chop_mask] = -np.inf
            preds = logits.argmax(dim=1)
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
        return loss_arr, preds_arr, labels_arr, logits_arr
    else:
        return loss_arr, preds_arr, labels_arr


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
