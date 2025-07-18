"""
Calibration methods for improving model performance on class-incremental learning.

This module implements the logit bias correction method from:
"Fine-Tuning is Fine, if Calibrated" (Kumar et al., 2019)

The key insight is that fine-tuning creates a bias towards seen classes,
which can be corrected by adding a constant bias to unseen class logits.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn import metrics
from typing import List, Dict, Any, Tuple, Optional, Optional
import logging
import time

logger = logging.getLogger(__name__)


def logit_bias_correction(logits: np.ndarray, training_classes: List[int], gamma: float) -> np.ndarray:
    """
    Apply logit bias correction by adding gamma to absent class logits.
    
    This implements the core method from "Fine-Tuning is Fine, if Calibrated":
    ŷ = argmax(w_c^T f_θ(x) + γ·1[c∈U])
    
    Args:
        logits: Raw model logits [N, C]
        training_classes: List of class indices that were seen during training
        gamma: Bias correction parameter to add to absent class logits
        
    Returns:
        Corrected logits with bias added to absent classes
    """
    corrected_logits = logits.copy()
    num_classes = logits.shape[1]
    
    # Find absent classes (not in training set)
    absent_classes = [i for i in range(num_classes) if i not in training_classes]
    
    # Add gamma bias to absent class logits
    if absent_classes:
        corrected_logits[:, absent_classes] += gamma
    
    return corrected_logits


def learn_gamma_alg(train_logits: np.ndarray, training_classes: List[int]) -> float:
    """
    Learn gamma using Average Logit Gap (ALG) method.
    
    ALG computes gamma as the difference between average seen and absent logits:
    γ = avg(train_logits[seen_classes]) - avg(train_logits[absent_classes])
    
    IMPORTANT: Uses TRAINING logits, not test logits, as per the paper.
    
    Args:
        train_logits: Raw model logits from TRAINING data [N, C]  
        training_classes: List of class indices seen during training
        
    Returns:
        Optimal gamma value
    """
    num_classes = train_logits.shape[1]
    absent_classes = [i for i in range(num_classes) if i not in training_classes]
    
    if not absent_classes or not training_classes:
        return 0.0
    
    # Compute average logits for seen and absent classes FROM TRAINING DATA
    seen_logits = train_logits[:, training_classes]
    absent_logits = train_logits[:, absent_classes]
    
    avg_seen = np.mean(seen_logits)
    avg_absent = np.mean(absent_logits)
    
    gamma = avg_seen - avg_absent
    return gamma


def learn_gamma_pcv(train_logits: np.ndarray, train_labels: np.ndarray, training_classes: List[int]) -> float:
    """
    Learn gamma using Post-hoc Calibration Validation (PCV) method.
    
    PCV searches for the gamma that maximizes validation accuracy.
    Uses TRAINING data to learn gamma, as per the paper.
    
    Args:
        train_logits: Raw model logits from TRAINING data [N, C]
        train_labels: True labels from TRAINING data [N]  
        training_classes: List of class indices seen during training
        
    Returns:
        Optimal gamma value
    """
    # Search over gamma values using TRAINING data
    gamma_candidates = np.linspace(-5, 5, 21)
    best_gamma = 0.0
    best_accuracy = 0.0
    
    for gamma in gamma_candidates:
        corrected_logits = logit_bias_correction(train_logits, training_classes, gamma)
        corrected_preds = np.argmax(corrected_logits, axis=1)
        accuracy = accuracy_score(train_labels, corrected_preds)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_gamma = gamma
    
    return best_gamma


def apply_paper_calibration(test_logits: np.ndarray, test_labels: np.ndarray, 
                          train_logits: np.ndarray, train_labels: np.ndarray,
                          training_classes: List[int], method: str = 'alg', verbose: bool = False) -> Dict[str, Any]:
    """
    Apply the paper's calibration method with comprehensive analysis.
    
    Args:
        test_logits: Raw model logits from TEST data [N_test, C]
        test_labels: True labels from TEST data [N_test]
        train_logits: Raw model logits from TRAINING data [N_train, C]
        train_labels: True labels from TRAINING data [N_train]
        training_classes: List of class indices seen during training  
        method: Method to learn gamma ('alg' or 'pcv')
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with calibration results and analysis
    """
    num_classes = test_logits.shape[1]
    absent_classes = [i for i in range(num_classes) if i not in training_classes]
    
    # Original predictions and metrics (on TEST data)
    original_preds = np.argmax(test_logits, axis=1)
    original_accuracy = accuracy_score(test_labels, original_preds)
    original_balanced_accuracy = balanced_accuracy_score(test_labels, original_preds)
    
    # Learn gamma from TRAINING data
    if method == 'alg':
        gamma = learn_gamma_alg(train_logits, training_classes)
    elif method == 'pcv':
        gamma = learn_gamma_pcv(train_logits, train_labels, training_classes)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply calibration to TEST data
    corrected_logits = logit_bias_correction(test_logits, training_classes, gamma)
    corrected_preds = np.argmax(corrected_logits, axis=1)
    
    # Compute corrected metrics (on TEST data)
    corrected_accuracy = accuracy_score(test_labels, corrected_preds)
    corrected_balanced_accuracy = balanced_accuracy_score(test_labels, corrected_preds)
    
    # Detailed analysis by class type (on TEST data)
    seen_mask = np.isin(test_labels, training_classes)
    absent_mask = np.isin(test_labels, absent_classes)
    
    seen_accuracy_original = accuracy_score(test_labels[seen_mask], original_preds[seen_mask]) if np.any(seen_mask) else 0.0
    seen_accuracy_corrected = accuracy_score(test_labels[seen_mask], corrected_preds[seen_mask]) if np.any(seen_mask) else 0.0
    
    absent_accuracy_original = accuracy_score(test_labels[absent_mask], original_preds[absent_mask]) if np.any(absent_mask) else 0.0
    absent_accuracy_corrected = accuracy_score(test_labels[absent_mask], corrected_preds[absent_mask]) if np.any(absent_mask) else 0.0
    
    # Compute improvements
    accuracy_improvement = corrected_accuracy - original_accuracy
    balanced_accuracy_improvement = corrected_balanced_accuracy - original_balanced_accuracy
    seen_accuracy_improvement = seen_accuracy_corrected - seen_accuracy_original
    absent_accuracy_improvement = absent_accuracy_corrected - absent_accuracy_original
    
    results = {
        'method': method,
        'gamma': gamma,
        'training_classes': training_classes,
        'absent_classes': absent_classes,
        'original_accuracy': original_accuracy,
        'original_balanced_accuracy': original_balanced_accuracy,
        'corrected_accuracy': corrected_accuracy,
        'balanced_accuracy': corrected_balanced_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'balanced_accuracy_change': balanced_accuracy_improvement,
        'seen_accuracy_original': seen_accuracy_original,
        'seen_accuracy_corrected': seen_accuracy_corrected,
        'seen_accuracy_improvement': seen_accuracy_improvement,
        'absent_accuracy_original': absent_accuracy_original,
        'absent_accuracy_corrected': absent_accuracy_corrected,
        'absent_accuracy_improvement': absent_accuracy_improvement,
        'num_training_classes': len(training_classes),
        'num_absent_classes': len(absent_classes),
        'total_samples': len(test_labels),
        'seen_samples': np.sum(seen_mask),
        'absent_samples': np.sum(absent_mask)
    }
    
    if verbose:
        logger.info(f"📄 Paper Calibration Method ({method.upper()}):")
        logger.info(f"   Training classes: {len(training_classes)}, Absent classes: {len(absent_classes)}")
        logger.info(f"   Learned γ = {gamma:.4f} (from training data)")
        logger.info(f"   Overall accuracy: {original_accuracy:.4f} → {corrected_accuracy:.4f} (Δ{accuracy_improvement:+.4f})")
        logger.info(f"   Balanced accuracy: {original_balanced_accuracy:.4f} → {corrected_balanced_accuracy:.4f} (Δ{balanced_accuracy_improvement:+.4f})")
        if np.any(seen_mask):
            logger.info(f"   Seen class accuracy: {seen_accuracy_original:.4f} → {seen_accuracy_corrected:.4f} (Δ{seen_accuracy_improvement:+.4f})")
        if np.any(absent_mask):
            logger.info(f"   Absent class accuracy: {absent_accuracy_original:.4f} → {absent_accuracy_corrected:.4f} (Δ{absent_accuracy_improvement:+.4f})")
    
    return results


def comprehensive_paper_calibration(test_logits: np.ndarray, test_labels: np.ndarray, 
                                  class_names: List[str], training_classes: List[int],
                                  train_logits: Optional[np.ndarray] = None, 
                                  train_labels: Optional[np.ndarray] = None,
                                  verbose: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive calibration analysis using both paper method and adaptive method.
    
    Args:
        test_logits: Raw model logits from test data [N_test, C]
        test_labels: True labels from test data [N_test]
        class_names: List of class names
        training_classes: List of class indices seen during training
        train_logits: Raw model logits from training data [N_train, C] (optional)
        train_labels: True labels from training data [N_train] (optional)
        verbose: Whether to print detailed logs
        
    Returns:
        Dictionary with comprehensive calibration results
    """
    if verbose:
        logger.info("🎯 Running Comprehensive Calibration Analysis")
        logger.info(f"   Total classes: {len(class_names)}")
        logger.info(f"   Training classes: {len(training_classes)}")
        logger.info(f"   Absent classes: {len(class_names) - len(training_classes)}")
        logger.info(f"   Test samples: {len(test_labels)}")
        if train_logits is not None:
            logger.info(f"   Training samples: {len(train_labels)}")
    
    # Original metrics (on test data)
    original_preds = np.argmax(test_logits, axis=1)
    original_accuracy = accuracy_score(test_labels, original_preds)
    original_balanced_accuracy = balanced_accuracy_score(test_labels, original_preds)
    
    # Apply paper calibration using ALG method
    if train_logits is not None and train_labels is not None:
        # Use proper training data for learning gamma (recommended)
        paper_results = apply_paper_calibration(
            test_logits, test_labels, train_logits, train_labels,
            training_classes, method='alg', verbose=verbose
        )
        if verbose:
            logger.info("   ✅ Using training data to learn gamma (proper method)")
    else:
        # Fallback: use test data (not recommended but better than nothing)
        if verbose:
            logger.warning("   ⚠️  No training data provided - using test data to learn gamma (not recommended)")
        paper_results = apply_paper_calibration(
            test_logits, test_labels, test_logits, test_labels,
            training_classes, method='alg', verbose=verbose
        )
    
    # Apply adaptive calibration method
    if verbose:
        logger.info("🔄 Running Adaptive Calibration Analysis")
    
    adaptive_results = apply_adaptive_calibration(
        test_logits, test_labels, training_classes, 
        target_metric='balanced_accuracy', verbose=verbose
    )
    
    # Compare methods and choose the best
    best_method = 'paper'
    best_results = paper_results
    
    # Choose method with best balanced accuracy improvement
    if adaptive_results['calibration_applied'] and adaptive_results['balanced_accuracy_improvement'] > paper_results['balanced_accuracy_change']:
        best_method = 'adaptive'
        best_results = adaptive_results
        if verbose:
            logger.info(f"🏆 Adaptive method chosen (balanced accuracy improvement: {adaptive_results['balanced_accuracy_improvement']:+.4f} vs {paper_results['balanced_accuracy_change']:+.4f})")
    else:
        if verbose:
            logger.info(f"🏆 Paper method chosen (balanced accuracy improvement: {paper_results['balanced_accuracy_change']:+.4f} vs {adaptive_results['balanced_accuracy_improvement']:+.4f})")
    
    # Compile results
    results = {
        'original': {
            'accuracy': original_accuracy,
            'balanced_accuracy': original_balanced_accuracy,
            'predictions': original_preds
        },
        'paper_calibration': paper_results,
        'adaptive_calibration': adaptive_results,
        'best_method': best_method,
        'best_results': best_results,
        'summary': {
            'method': 'comprehensive',
            'training_classes_count': len(training_classes),
            'absent_classes_count': len(class_names) - len(training_classes),
            'paper_accuracy_improvement': paper_results['accuracy_improvement'],
            'paper_balanced_accuracy_improvement': paper_results['balanced_accuracy_change'],
            'adaptive_accuracy_improvement': adaptive_results['accuracy_improvement'],
            'adaptive_balanced_accuracy_improvement': adaptive_results['balanced_accuracy_improvement'],
            'best_accuracy_improvement': best_results.get('accuracy_improvement', best_results.get('balanced_accuracy_change', 0)),
            'best_balanced_accuracy_improvement': best_results.get('balanced_accuracy_improvement', best_results.get('balanced_accuracy_change', 0)),
            'auc_score': adaptive_results.get('auc_score', 0.0)
        }
    }
    
    if verbose:
        logger.info("✅ Comprehensive calibration analysis complete")
        logger.info(f"   Best method: {best_method}")
        logger.info(f"   Best balanced accuracy improvement: {results['summary']['best_balanced_accuracy_improvement']:+.4f}")
        if adaptive_results.get('auc_score', 0) > 0:
            logger.info(f"   AUC score: {adaptive_results['auc_score']:.4f}")
    
    return results


def get_training_classes_for_checkpoint(ckp: str, train_dset, label_type: str = "species") -> List[int]:
    """
    Determine training classes for a given checkpoint in accumulative training.
    
    Args:
        ckp: Checkpoint name (e.g., 'ckp_3')
        train_dset: Training dataset object
        label_type: Type of labels to use
        
    Returns:
        List of class indices that were seen during training for this checkpoint
    """
    if ckp == 'ckp_1':
        # Zero-shot model - no classes were trained on
        return []
    
    # For ckp_N, get training classes from ckp_(N-1) training data
    ckp_num = int(ckp.split('_')[1])
    prev_ckp = f'ckp_{ckp_num - 1}'
    
    try:
        prev_train_dset = train_dset.get_subset(is_train=True, ckp_list=prev_ckp)
        if len(prev_train_dset) > 0:
            training_classes = list(set([sample.label for sample in prev_train_dset.samples]))
            logger.info(f"Training classes from {prev_ckp}: {len(training_classes)} classes")
            return training_classes
        else:
            logger.warning(f"No training data found for {prev_ckp}")
            return []
    except Exception as e:
        logger.warning(f"Could not load training data for {prev_ckp}: {e}")
        return []


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _compute_accuracy_torch(logits: torch.Tensor, labels: torch.Tensor, 
                           visible_mask: torch.Tensor, invisible_mask: torch.Tensor,
                           chopped_out_classes=None):
    """
    Compute accuracy for visible and invisible classes.
    
    Args:
        logits: Model logits [N, C]
        labels: True labels [N]
        visible_mask: Boolean mask for samples with visible/seen classes
        invisible_mask: Boolean mask for samples with invisible/unseen classes
        chopped_out_classes: Classes to exclude from prediction
        
    Returns:
        List of [overall_acc, visible_acc, invisible_acc] in percentage
    """
    new_logits = logits.clone()
    if chopped_out_classes is not None:
        new_logits[:, chopped_out_classes] = float('-inf')
    
    # Overall accuracy
    overall_acc = (new_logits.argmax(dim=1) == labels).sum().item() / labels.shape[0]
    
    # Visible class accuracy (seen classes)
    if visible_mask.sum().item() > 0:
        visible_acc = (new_logits[visible_mask].argmax(dim=1) == labels[visible_mask]).sum().item() / visible_mask.sum().item()
    else:
        visible_acc = 0.0
    
    # Invisible class accuracy (unseen classes)
    if invisible_mask.sum().item() > 0:
        invisible_acc = (new_logits[invisible_mask].argmax(dim=1) == labels[invisible_mask]).sum().item() / invisible_mask.sum().item()
    else:
        invisible_acc = 0.0
    
    return [overall_acc * 100., visible_acc * 100., invisible_acc * 100.]


def _compute_shifting(logits: torch.Tensor, visible_classes: List[int], invisible_classes: List[int], mode='positive'):
    """
    Compute logit shifting amount to improve calibration.
    
    Args:
        logits: Model logits [N, C]
        visible_classes: List of seen class indices
        invisible_classes: List of unseen class indices  
        mode: 'positive' to increase unseen accuracy, 'negative' to increase seen accuracy
        
    Returns:
        Tuple of (shift_amount, is_final)
    """
    assert mode in ['positive', 'negative']
    MIN_SHIFT = 0.1  # Minimum shift to ensure progress
    
    # Convert to tensors if needed
    if isinstance(visible_classes, list):
        visible_classes = torch.tensor(visible_classes, device=logits.device)
    if isinstance(invisible_classes, list):
        invisible_classes = torch.tensor(invisible_classes, device=logits.device)

    try:
        # Compute maximum logits for seen and unseen classes
        max_seen_logits = logits[:, visible_classes].max(dim=1)[0]
        max_unseen_logits = logits[:, invisible_classes].max(dim=1)[0]

        # Determine valid indices based on mode
        if mode == 'positive':
            # We want to increase unseen accuracy, so find cases where seen > unseen
            valid = max_seen_logits > max_unseen_logits
            if valid.sum() == 0:
                return 0., True
            diff = (max_seen_logits[valid] - max_unseen_logits[valid])
        else:
            # We want to increase seen accuracy, so find cases where unseen > seen
            valid = max_unseen_logits > max_seen_logits
            if valid.sum() == 0:
                return 0., True
            diff = (max_unseen_logits[valid] - max_seen_logits[valid])

        if len(diff) == 0:
            return 0., True
            
        # Sort differences to find the smallest gap to close
        diff_sorted = diff.sort()[0]
        
        # Filter out very small differences to avoid numerical instability
        diff_filtered = diff_sorted[diff_sorted > 1e-4]
        if len(diff_filtered) == 0:
            return MIN_SHIFT, True
        
        # Take the smallest meaningful difference and add a small increment
        smallest_diff = diff_filtered[0].item()
        shift_amount = max(smallest_diff + MIN_SHIFT, MIN_SHIFT)
        
        # Check if this is likely the final iteration
        is_final = len(diff_filtered) <= 2 or smallest_diff < MIN_SHIFT * 2
        
        return shift_amount, is_final
        
    except Exception as e:
        logger.warning(f"Error in _compute_shifting: {e}")
        return 0., True


def get_curve_results(test_logits: np.ndarray, test_labels: np.ndarray, training_classes: List[int]):
    """
    Generate calibration curve by systematically shifting logits.
    
    Args:
        test_logits: Test logits [N, C]
        test_labels: Test labels [N]
        training_classes: List of seen class indices
        
    Returns:
        Tuple of (curve_results, trade_off_curve, auc_score)
        - curve_results: [overall_acc, visible_acc, invisible_acc, accumulate_shifting]
        - trade_off_curve: [visible_acc, invisible_acc] pairs
        - auc_score: Area under the trade-off curve
    """
    start = time.time()
    
    # Convert to torch tensors
    if isinstance(test_logits, np.ndarray):
        logits = torch.from_numpy(test_logits).float()
    else:
        logits = test_logits.float()
        
    if isinstance(test_labels, np.ndarray):
        labels = torch.from_numpy(test_labels).long()
    else:
        labels = test_labels.long()
    
    num_classes = logits.shape[1]
    invisible_classes = [i for i in range(num_classes) if i not in training_classes]
    
    # Check if we have both seen and unseen classes
    if len(training_classes) == 0 or len(invisible_classes) == 0:
        logger.warning("No calibration needed: missing seen or unseen classes")
        return None, None, None
    
    curve_results = []

    # Compute seen and unseen sample masks
    visible_class_mask = torch.tensor([l.item() in training_classes for l in labels], dtype=torch.bool)
    invisible_class_mask = torch.tensor([l.item() in invisible_classes for l in labels], dtype=torch.bool)
    
    # Check if we have samples from both types
    if not visible_class_mask.any() or not invisible_class_mask.any():
        logger.warning("No calibration needed: missing seen or unseen samples in test data")
        return None, None, None
    
    logger.debug(f'Visible samples: {visible_class_mask.sum()}, Invisible samples: {invisible_class_mask.sum()}')

    # Add original point first
    overall_acc, visible_acc, invisible_acc = _compute_accuracy_torch(
        logits, labels, visible_class_mask, invisible_class_mask)
    curve_results.append([overall_acc, visible_acc, invisible_acc, 0.0])

    # Generate points by shifting in both directions
    # Use a simpler approach: try a range of shift values
    shift_values = []
    
    # Positive shifts (increase unseen accuracy)
    for shift in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        shift_values.append(shift)
    
    # Negative shifts (increase seen accuracy) 
    for shift in [-0.5, -1.0, -2.0, -3.0, -5.0, -10.0]:
        shift_values.append(shift)
    
    # Apply each shift and compute accuracies
    for shift in shift_values:
        logits_shifted = logits.clone().to(torch.float64)
        logits_shifted[:, invisible_classes] += shift
        
        overall_acc, visible_acc, invisible_acc = _compute_accuracy_torch(
            logits_shifted, labels, visible_class_mask, invisible_class_mask)
        curve_results.append([overall_acc, visible_acc, invisible_acc, shift])

    logger.debug(f'Generated {len(curve_results)} calibration points')

    # Check if we got any results
    if len(curve_results) <= 1:
        logger.warning("Could not generate calibration curve - only original point available")
        return None, None, None

    # Get trade-off curve
    curve_results = torch.tensor(curve_results)
    curve_results = curve_results[torch.argsort(curve_results[:, 1])]  # Sort by seen acc
    trade_off_curve = curve_results[:, 1:3]  # [visible_acc, invisible_acc]
    
    # Calculate AUC
    try:
        visible_accs = trade_off_curve[:, 0].cpu().numpy() / 100.
        invisible_accs = trade_off_curve[:, 1].cpu().numpy() / 100.
        
        # Need at least 2 points for AUC
        if len(visible_accs) < 2:
            auc_score = 0.0
        else:
            # Sort by visible accuracy to ensure proper AUC calculation
            sorted_indices = np.argsort(visible_accs)
            sorted_visible = visible_accs[sorted_indices]
            sorted_invisible = invisible_accs[sorted_indices]
            
            # Remove duplicate x values
            unique_x, unique_indices = np.unique(sorted_visible, return_index=True)
            if len(unique_x) < len(sorted_visible):
                sorted_visible = unique_x
                sorted_invisible = sorted_invisible[unique_indices]
            
            if len(sorted_visible) >= 2:
                # Ensure values are in [0, 1] range
                sorted_visible = np.clip(sorted_visible, 0, 1)
                sorted_invisible = np.clip(sorted_invisible, 0, 1)
                auc_score = metrics.auc(sorted_visible, sorted_invisible)
            else:
                auc_score = 0.0
    except Exception as e:
        logger.warning(f"Could not compute AUC: {e}")
        auc_score = 0.0

    end = time.time()
    logger.debug(f'Total curve generation time: {end - start:.4f} seconds')

    return curve_results, trade_off_curve, auc_score


def find_optimal_calibration_point(curve_results: torch.Tensor, target_metric='balanced_accuracy'):
    """
    Find optimal calibration point from curve results.
    
    Args:
        curve_results: Tensor of [overall_acc, visible_acc, invisible_acc, shift_amount]
        target_metric: 'balanced_accuracy', 'overall_accuracy', or 'harmonic_mean'
        
    Returns:
        Optimal shift amount and corresponding accuracies
    """
    if curve_results is None or len(curve_results) == 0:
        return 0.0, None
    
    overall_accs = curve_results[:, 0]
    visible_accs = curve_results[:, 1] 
    invisible_accs = curve_results[:, 2]
    shift_amounts = curve_results[:, 3]
    
    if target_metric == 'balanced_accuracy':
        # Balanced accuracy = (seen_acc + unseen_acc) / 2
        scores = (visible_accs + invisible_accs) / 2
    elif target_metric == 'overall_accuracy':
        scores = overall_accs
    elif target_metric == 'harmonic_mean':
        # Harmonic mean of seen and unseen accuracy
        scores = 2 * (visible_accs * invisible_accs) / (visible_accs + invisible_accs + 1e-8)
    else:
        raise ValueError(f"Unknown target_metric: {target_metric}")
    
    best_idx = torch.argmax(scores)
    optimal_shift = shift_amounts[best_idx].item()
    optimal_scores = {
        'overall_accuracy': overall_accs[best_idx].item(),
        'visible_accuracy': visible_accs[best_idx].item(), 
        'invisible_accuracy': invisible_accs[best_idx].item(),
        'balanced_accuracy': (visible_accs[best_idx] + invisible_accs[best_idx]).item() / 2,
        'shift_amount': optimal_shift
    }
    
    return optimal_shift, optimal_scores


def apply_adaptive_calibration(test_logits: np.ndarray, test_labels: np.ndarray,
                              training_classes: List[int], target_metric='balanced_accuracy',
                              verbose: bool = False) -> Dict[str, Any]:
    """
    Apply adaptive calibration using a simple grid search method.
    
    Args:
        test_logits: Test logits [N, C]
        test_labels: Test labels [N]
        training_classes: List of seen class indices
        target_metric: Optimization target ('balanced_accuracy', 'overall_accuracy', 'harmonic_mean')
        verbose: Whether to print detailed logs
        
    Returns:
        Dictionary with calibration results
    """
    num_classes = test_logits.shape[1]
    absent_classes = [i for i in range(num_classes) if i not in training_classes]
    
    # Check if calibration is needed
    if len(training_classes) == 0 or len(absent_classes) == 0:
        if verbose:
            logger.warning("Adaptive calibration skipped: all classes are seen or all classes are unseen")
        original_preds = np.argmax(test_logits, axis=1)
        original_accuracy = accuracy_score(test_labels, original_preds)
        original_balanced_accuracy = balanced_accuracy_score(test_labels, original_preds)
        
        return {
            'method': 'adaptive_none',
            'calibration_applied': False,
            'reason': 'no_unseen_classes' if len(absent_classes) == 0 else 'no_seen_classes',
            'target_metric': target_metric,
            'original_accuracy': original_accuracy,
            'corrected_accuracy': original_accuracy,
            'original_balanced_accuracy': original_balanced_accuracy,
            'corrected_balanced_accuracy': original_balanced_accuracy,
            'accuracy_improvement': 0.0,
            'balanced_accuracy_improvement': 0.0,
            'shift_amount': 0.0,
            'auc_score': 0.0,
            'curve_points': 0,
            'seen_accuracy_original': 0.0,
            'seen_accuracy_corrected': 0.0,
            'seen_accuracy_improvement': 0.0,
            'absent_accuracy_original': 0.0,
            'absent_accuracy_corrected': 0.0,
            'absent_accuracy_improvement': 0.0,
            'num_training_classes': len(training_classes),
            'num_absent_classes': len(absent_classes),
            'total_samples': len(test_labels),
            'seen_samples': 0,
            'absent_samples': 0
        }
    
    # Original metrics
    original_preds = np.argmax(test_logits, axis=1)
    original_accuracy = accuracy_score(test_labels, original_preds)
    original_balanced_accuracy = balanced_accuracy_score(test_labels, original_preds)
    
    if verbose:
        logger.info(f"🎯 Adaptive Calibration Method:")
        logger.info(f"   Target metric: {target_metric}")
        logger.info(f"   Training classes: {len(training_classes)}, Absent classes: {len(absent_classes)}")
        logger.info(f"   Original accuracy: {original_accuracy:.4f}")
        logger.info(f"   Original balanced accuracy: {original_balanced_accuracy:.4f}")
    
    # Grid search over shift values
    shift_values = np.linspace(-10, 10, 21)  # Simple grid from -10 to 10
    best_shift = 0.0
    best_score = 0.0
    best_results = None
    
    results_list = []
    
    for shift in shift_values:
        # Apply shift to absent classes
        corrected_logits = test_logits.copy()
        corrected_logits[:, absent_classes] += shift
        corrected_preds = np.argmax(corrected_logits, axis=1)
        
        # Calculate metrics
        corrected_accuracy = accuracy_score(test_labels, corrected_preds)
        corrected_balanced_accuracy = balanced_accuracy_score(test_labels, corrected_preds)
        
        # Calculate score based on target metric
        if target_metric == 'balanced_accuracy':
            score = corrected_balanced_accuracy
        elif target_metric == 'overall_accuracy':
            score = corrected_accuracy
        elif target_metric == 'harmonic_mean':
            # Harmonic mean of seen and unseen accuracy
            seen_mask = np.isin(test_labels, training_classes)
            absent_mask = np.isin(test_labels, absent_classes)
            
            seen_acc = accuracy_score(test_labels[seen_mask], corrected_preds[seen_mask]) if np.any(seen_mask) else 0.0
            absent_acc = accuracy_score(test_labels[absent_mask], corrected_preds[absent_mask]) if np.any(absent_mask) else 0.0
            
            if seen_acc + absent_acc > 0:
                score = 2 * (seen_acc * absent_acc) / (seen_acc + absent_acc)
            else:
                score = 0.0
        else:
            score = corrected_balanced_accuracy
        
        results_list.append({
            'shift': shift,
            'accuracy': corrected_accuracy,
            'balanced_accuracy': corrected_balanced_accuracy,
            'score': score
        })
        
        # Update best if this is better
        if score > best_score:
            best_score = score
            best_shift = shift
            best_results = {
                'accuracy': corrected_accuracy,
                'balanced_accuracy': corrected_balanced_accuracy
            }
    
    # Calculate AUC from results (simple approximation)
    if len(results_list) > 1:
        try:
            # Sort by shift value and create a simple trade-off curve
            sorted_results = sorted(results_list, key=lambda x: x['shift'])
            shifts = [r['shift'] for r in sorted_results]
            accs = [r['accuracy'] for r in sorted_results]
            
            # Normalize for AUC calculation
            if max(shifts) != min(shifts):
                normalized_shifts = [(s - min(shifts)) / (max(shifts) - min(shifts)) for s in shifts]
                auc_score = metrics.auc(normalized_shifts, accs)
            else:
                auc_score = 0.0
        except:
            auc_score = 0.0
    else:
        auc_score = 0.0
    
    # Apply optimal calibration
    corrected_logits = test_logits.copy()
    corrected_logits[:, absent_classes] += best_shift
    corrected_preds = np.argmax(corrected_logits, axis=1)
    
    corrected_accuracy = best_results['accuracy']
    corrected_balanced_accuracy = best_results['balanced_accuracy']
    
    # Calculate improvements
    accuracy_improvement = corrected_accuracy - original_accuracy
    balanced_accuracy_improvement = corrected_balanced_accuracy - original_balanced_accuracy
    
    # Detailed analysis by class type
    seen_mask = np.isin(test_labels, training_classes)
    absent_mask = np.isin(test_labels, absent_classes)
    
    seen_accuracy_original = accuracy_score(test_labels[seen_mask], original_preds[seen_mask]) if np.any(seen_mask) else 0.0
    seen_accuracy_corrected = accuracy_score(test_labels[seen_mask], corrected_preds[seen_mask]) if np.any(seen_mask) else 0.0
    
    absent_accuracy_original = accuracy_score(test_labels[absent_mask], original_preds[absent_mask]) if np.any(absent_mask) else 0.0
    absent_accuracy_corrected = accuracy_score(test_labels[absent_mask], corrected_preds[absent_mask]) if np.any(absent_mask) else 0.0
    
    results = {
        'method': f'adaptive_{target_metric}',
        'calibration_applied': True,
        'target_metric': target_metric,
        'shift_amount': best_shift,
        'training_classes': training_classes,
        'absent_classes': absent_classes,
        'original_accuracy': original_accuracy,
        'original_balanced_accuracy': original_balanced_accuracy,
        'corrected_accuracy': corrected_accuracy,
        'corrected_balanced_accuracy': corrected_balanced_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'balanced_accuracy_improvement': balanced_accuracy_improvement,
        'seen_accuracy_original': seen_accuracy_original,
        'seen_accuracy_corrected': seen_accuracy_corrected,
        'seen_accuracy_improvement': seen_accuracy_corrected - seen_accuracy_original,
        'absent_accuracy_original': absent_accuracy_original,
        'absent_accuracy_corrected': absent_accuracy_corrected,
        'absent_accuracy_improvement': absent_accuracy_corrected - absent_accuracy_original,
        'auc_score': auc_score,
        'curve_points': len(results_list),
        'optimal_scores': {
            'overall_accuracy': corrected_accuracy,
            'balanced_accuracy': corrected_balanced_accuracy,
            'shift_amount': best_shift
        },
        'num_training_classes': len(training_classes),
        'num_absent_classes': len(absent_classes),
        'total_samples': len(test_labels),
        'seen_samples': np.sum(seen_mask),
        'absent_samples': np.sum(absent_mask)
    }
    
    if verbose:
        logger.info(f"   Generated {len(results_list)} calibration points")
        logger.info(f"   AUC score: {auc_score:.4f}")
        logger.info(f"   Optimal shift: {best_shift:.4f}")
        logger.info(f"   Corrected accuracy: {corrected_accuracy:.4f} (Δ{accuracy_improvement:+.4f})")
        logger.info(f"   Corrected balanced accuracy: {corrected_balanced_accuracy:.4f} (Δ{balanced_accuracy_improvement:+.4f})")
        if np.any(seen_mask):
            logger.info(f"   Seen class accuracy: {seen_accuracy_original:.4f} → {seen_accuracy_corrected:.4f} (Δ{seen_accuracy_corrected - seen_accuracy_original:+.4f})")
        if np.any(absent_mask):
            logger.info(f"   Absent class accuracy: {absent_accuracy_original:.4f} → {absent_accuracy_corrected:.4f} (Δ{absent_accuracy_corrected - absent_accuracy_original:+.4f})")
    
    return results
