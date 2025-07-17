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
from typing import List, Dict, Any, Tuple, Optional, Optional
import logging

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
    Run comprehensive calibration analysis using only the paper's method.
    
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
        logger.info("🎯 Running Paper Calibration Analysis")
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
    
    # Compile results
    results = {
        'original': {
            'accuracy': original_accuracy,
            'balanced_accuracy': original_balanced_accuracy,
            'predictions': original_preds
        },
        'paper_calibration': paper_results,
        'summary': {
            'method': 'paper_only',
            'training_classes_count': len(training_classes),
            'absent_classes_count': len(class_names) - len(training_classes),
            'accuracy_improvement': paper_results['accuracy_improvement'],
            'balanced_accuracy_improvement': paper_results['balanced_accuracy_change'],
            'absent_accuracy_improvement': paper_results['absent_accuracy_improvement']
        }
    }
    
    if verbose:
        logger.info("✅ Paper calibration analysis complete")
        logger.info(f"   Best improvement: Δ_accuracy={paper_results['accuracy_improvement']:+.4f}, Δ_balanced_accuracy={paper_results['balanced_accuracy_change']:+.4f}")
    
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
