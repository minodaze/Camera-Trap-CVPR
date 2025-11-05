import matplotlib.pyplot as plt

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
from plot.plot_features import plot_features
from plot.plot_text_F import plot_text_F

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
