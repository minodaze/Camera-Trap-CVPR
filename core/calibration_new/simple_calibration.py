# inputs: seen classes [1, 0 ,0], logits [# images, total classes] 
# output: calibrated logits

import numpy as np
import torch

def run_simple_calibration(eval_dset, ckp_eval_dset, labels_arr_cal, logits_arr_cal, class_names):
    
    # Get seen classes for current checkpoint
    current_seen_classes = set()
    # cls_match = eval_dset.class_match
    for sample in ckp_eval_dset.samples:
        current_seen_classes.add(sample.label)
    current_seen_classes_list = sorted(list(current_seen_classes))
    
    # Create seen_classes mask
    seen_classes_mask = np.zeros(len(class_names), dtype=int)
    # import pdb; pdb.set_trace()  # Debugging breakpoint to inspect seen_classes_mask
    
    for cls_idx, cls_name in enumerate(current_seen_classes_list):
        seen_classes_mask[cls_idx] = 1
    
    # Use eval function with return_logits=True to get logits
    
    # loss_arr_cal, preds_arr_cal, labels_arr_cal, logits_arr_cal = eval(
    #     classifier, cl_eval_loader, device, 
    #     chop_head=common_config['chop_head'], 
    #     return_logits=True
    # )
    
    if verify_calibration_inputs(seen_classes_mask, logits_arr_cal):
        
        calibrated_logits_list = simple_calibration(seen_classes_mask, logits_arr_cal)
        
        if calibrated_logits_list:
            current_ckp_results = evaluate_calibrated_logits(
                calibrated_logits_list, labels_arr_cal, class_names
            )
            
            # Find the best calibration result based on current checkpoint balanced accuracy only
            best_calibration = max(current_ckp_results, key=lambda x: x['balanced_accuracy'])
            best_matrix_idx = best_calibration['matrix_index']
            
            # print(f"🎯 Best calibration result for current checkpoint {ckp}:")
            # print(f"  Matrix index: {best_matrix_idx}")
            # print(f"  Current ckp accuracy: {best_calibration['accuracy']:.4f}")
            # print(f"  Current ckp balanced accuracy: {best_calibration['balanced_accuracy']:.4f}")
            
            # Store calibration results for current checkpoint
            # calibration_results = {
            #     'enabled': True,
            #     'current_checkpoint': ckp,
            #     'mode': 'inference',
            #     'seen_classes_count': int(seen_classes_mask.sum()),
            #     'unseen_classes_count': int(len(seen_classes_mask) - seen_classes_mask.sum()),
            #     'num_calibrated_matrices': len(calibrated_logits_list),
            #     'current_ckp_results': current_ckp_results,
            #     'best_calibration': best_calibration,
            #     'best_matrix_index': best_matrix_idx
            # }
            
            # Store the best balanced accuracy for this checkpoint in global list
            # best_calibrated_balanced_acc_list.append(best_calibration['balanced_accuracy'])
            
            # # Update final_eval_results with calibration info
            # final_eval_results[ckp]['calibration'] = calibration_results
            # final_eval_results[ckp]['best_calibrated_balanced_accuracy'] = best_calibration['balanced_accuracy']
                
    return best_calibration


def simple_calibration(seen_classes, logits):
    """
    Simple calibration function that adjusts logits based on seen/unseen classes.
    
    Args:
        seen_classes: binary array of shape [total_classes] where 1 indicates seen class, 0 indicates unseen
        logits: tensor/array of shape [num_images, total_classes] containing raw logits
        
    Returns:
        calibrated_logits_list: list of calibrated logits matrices, one for each calibration value
    """
    # Verify inputs first
    if not verify_calibration_inputs(seen_classes, logits):
        print("ERROR: Calibration input verification failed!")
        return []
    
    # Convert to numpy arrays for easier manipulation
    if torch.is_tensor(seen_classes):
        seen_classes = seen_classes.cpu().numpy()
    if torch.is_tensor(logits):
        logits = logits.cpu().numpy()
    
    seen_classes = np.array(seen_classes, dtype=bool)
    logits = np.array(logits)
    
    num_images, total_classes = logits.shape
    
    # Create masks for seen and unseen classes
    seen_mask = seen_classes
    unseen_mask = ~seen_classes
    
    # List to store S - T values for each datapoint
    calibration_values = []
    
    # For each datapoint (row in logits)
    for i in range(num_images):
        current_logits = logits[i]  # Shape: [total_classes]
        
        # Get max logit for seen classes (S)
        seen_logits = current_logits[seen_mask]
        if len(seen_logits) > 0:
            S = np.max(seen_logits)
        else:
            S = -np.inf  # If no seen classes, set to negative infinity
        
        # Get max logit for unseen classes (T)
        unseen_logits = current_logits[unseen_mask]
        if len(unseen_logits) > 0:
            T = np.max(unseen_logits)
        else:
            T = -np.inf  # If no unseen classes, set to negative infinity
        
        # Calculate calibration value k = S - T
        k = S - T
        calibration_values.append(k)
    average_calibration_factor = np.mean(calibration_values)
    calibration_values.append(average_calibration_factor)
    # Apply calibration: for each k in calibration_values, create a calibrated logits matrix
    calibrated_logits_list = []
    
    # print(f"🔧 Applying calibration with {len(calibration_values)} different k values...")


    for k_idx, k in enumerate(calibration_values):
        # Add small epsilon to k
        k_adjusted = k + 0.00000001
        
        # Apply this k to all rows in the logits matrix
        calibrated_logits = logits.copy()  # Make a copy of the entire matrix
        
        # Add k_adjusted to unseen class positions for ALL rows
        
        calibrated_logits[:, unseen_mask] += k_adjusted

        calibrated_logits_list.append(calibrated_logits)
        
        # print(f"  ✓ Matrix {k_idx}: k={k:.6f} + ε = {k_adjusted:.6f}")
    
    # print(f"✅ Calibration completed: generated {len(calibrated_logits_list)} calibrated matrices")
    
    # Verify output dimensions
    for i, cal_logits in enumerate(calibrated_logits_list):
        if cal_logits.shape != logits.shape:
            print(f"❌ ERROR: Calibrated matrix {i} shape mismatch: {cal_logits.shape} != {logits.shape}")
            return []
    
    if len(calibrated_logits_list) != logits.shape[0] + 1:
        print(f"Error, expected {logits.shape[0] + 1} matrices, got {len(calibrated_logits_list)}")
        return []
    
    # print(f"✓ All calibrated matrices verified: shape {logits.shape}")
    
    return calibrated_logits_list
    


def verify_calibration_inputs(seen_classes, logits):
    """
    Verify the inputs for calibration function.
    
    Args:
        seen_classes: binary array indicating seen classes
        logits: logits matrix
    
    Returns:
        bool: True if inputs are valid, False otherwise
    """
    try:
        # Convert to numpy for verification
        if torch.is_tensor(seen_classes):
            seen_classes = seen_classes.cpu().numpy()
        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()
        
        seen_classes = np.array(seen_classes)
        logits = np.array(logits)
        
        # Check dimensions
        if len(seen_classes.shape) != 1:
            print(f"ERROR: seen_classes should be 1D, got shape {seen_classes.shape}")
            return False
        
        if len(logits.shape) != 2:
            print(f"ERROR: logits should be 2D, got shape {logits.shape}")
            return False
        
        # Check compatibility
        if seen_classes.shape[0] != logits.shape[1]:
            print(f"ERROR: seen_classes length ({seen_classes.shape[0]}) != logits classes ({logits.shape[1]})")
            return False
        
        # Check if there are both seen and unseen classes
        n_seen = np.sum(seen_classes)
        n_unseen = len(seen_classes) - n_seen
        
        if n_seen == 0:
            print("WARNING: No seen classes found")
            return False
        
        if n_unseen == 0:
            print("WARNING: No unseen classes found, calibration may not be meaningful")
        
        # print(f"✓ Calibration inputs verified: {logits.shape[0]} images, {logits.shape[1]} classes")
        print(f"✓ Seen classes: {n_seen}, Unseen classes: {n_unseen}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in calibration input verification: {e}")
        return False

def evaluate_calibrated_logits(calibrated_logits_list, test_labels, class_names):
    """
    Evaluate each calibrated logits matrix and return metrics.
    
    Args:
        calibrated_logits_list: List of calibrated logits matrices
        test_labels: True labels for the test data
        class_names: List of class names
    
    Returns:
        list: List of evaluation results for each calibrated matrix
    """
    results = []
    
    # print(f"🔍 Evaluating {len(calibrated_logits_list)} calibrated logits matrices...")
    
    for i, calibrated_logits in enumerate(calibrated_logits_list):
        # Get predictions from calibrated logits
        preds = np.argmax(calibrated_logits, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(preds == test_labels)
        
        # Calculate balanced accuracy
        n_classes = len(class_names)
        acc_per_class = []
        for cls_idx in range(n_classes):
            mask = test_labels == cls_idx
            if mask.sum() > 0:
                class_acc = np.mean(preds[mask] == test_labels[mask])
                acc_per_class.append(class_acc)
        
        balanced_acc = np.mean(acc_per_class) if acc_per_class else 0.0
        
        result = {
            'matrix_index': i,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'num_samples': len(test_labels)
        }
        
        results.append(result)
        # print(f"  Matrix {i}: acc={accuracy:.4f}, balanced_acc={balanced_acc:.4f}")
    
    return results

def find_best_calibration(evaluation_results):
    """
    Find the best calibration result based on balanced accuracy.
    
    Args:
        evaluation_results: List of evaluation results
    
    Returns:
        dict: Best calibration result
    """
    if not evaluation_results:
        return None
    
    best_result = max(evaluation_results, key=lambda x: x['balanced_accuracy'])
    
    print(f"🎯 Best calibration found:")
    print(f"  Matrix index: {best_result['matrix_index']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")
    
    return best_result

# Example usage:
if __name__ == "__main__":
    # Example with 3 classes, 2 images
    seen_classes = np.array([1, 0, 0])  # Only class 0 is seen
    logits = np.array([
        [2.0, 1.5, 0.8],  # Image 1 logits
        [1.2, 2.1, 1.0]   # Image 2 logits
    ])
    
    calibrated = simple_calibration(seen_classes, logits)
    
    print("Original logits:")
    print(logits)
    print("\nCalibration values (S - T):")
    
    # Show the calibration process step by step
    seen_mask = seen_classes.astype(bool)
    unseen_mask = ~seen_mask
    
    for i in range(len(logits)):
        current_logits = logits[i]
        S = np.max(current_logits[seen_mask]) if np.any(seen_mask) else -np.inf
        T = np.max(current_logits[unseen_mask]) if np.any(unseen_mask) else -np.inf
        k = S - T
        print(f"Image {i+1}: S={S:.3f}, T={T:.3f}, k=S-T={k:.3f}")
    
    print("\nCalibrated logits matrices:")
    for i, cal_logits in enumerate(calibrated):
        print(f"Using k from Image {i+1} (k={calibrated[0][0, unseen_mask][0] - 0.00000001:.3f} + ε):")
        print(cal_logits)
        print()
        
    print(f"Seen classes mask: {seen_classes}")
    print(f"Unseen classes mask: {~seen_classes.astype(bool)}")
    print(f"Number of calibrated matrices: {len(calibrated)}")
    print(f"Each matrix shape: {calibrated[0].shape}")
