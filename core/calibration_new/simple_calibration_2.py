import numpy as np
import torch

def run_simple_calibration(eval_dset,
                           ckp_eval_dset,
                           labels_arr_cal,
                           logits_arr_cal,
                           class_names,
                           gamma_min=-10.0,
                           gamma_max=10.0,
                           num_steps=201):
    """
    CHEATING ORACLE CALIBRATION:
    - Uses labels_arr_cal + logits_arr_cal for both searching γ and reporting performance.
    - Upper bound for any method that applies a single global γ to all unseen classes.
    """

    # 1) Get seen classes for current checkpoint
    current_seen_classes = set()
    for sample in ckp_eval_dset.samples:
        current_seen_classes.add(sample.label)
    current_seen_classes_list = sorted(list(current_seen_classes))

    # 2) Build seen_classes_mask correctly (index by class_id)
    seen_classes_mask = np.zeros(len(class_names), dtype=int)
    for cls_id in current_seen_classes_list:
        seen_classes_mask[cls_id] = 1

    # 3) Verify inputs
    if not verify_calibration_inputs(seen_classes_mask, logits_arr_cal):
        print("❌ Calibration inputs invalid, aborting.")
        return None

    # 4) Generate calibrated logits for many γ candidates (cheating grid search)
    calibrated_logits_list, gamma_candidates = simple_calibration(
        seen_classes_mask,
        logits_arr_cal,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        num_steps=num_steps,
    )

    # 5) Evaluate all candidates using the SAME labels (cheating upper bound)
    current_ckp_results = evaluate_calibrated_logits(
        calibrated_logits_list,
        labels_arr_cal,
        class_names
    )

    # 6) Find best γ by balanced accuracy
    best_calibration = find_best_calibration(current_ckp_results)
    if best_calibration is None:
        return None

    best_matrix_idx = best_calibration['matrix_index']
    best_gamma = gamma_candidates[best_matrix_idx]

    # Attach γ to result for convenience
    best_calibration['gamma'] = float(best_gamma)

    print("🎯 CHEATING ORACLE CALIBRATION (GLOBAL γ)")
    print(f"  Best γ index: {best_matrix_idx}")
    print(f"  Best γ value: {best_gamma:.4f}")
    print(f"  Accuracy: {best_calibration['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_calibration['balanced_accuracy']:.4f}")

    return best_calibration


def simple_calibration(seen_classes,
                       logits,
                       gamma_min=-5.0,
                       gamma_max=5.0,
                       num_steps=201):
    """
    CHEATING ORACLE CALIBRATION:
    - Try a dense grid of γ values.
    - For each γ, add γ to all unseen-class logits globally.
    - Returns:
        - list of calibrated logits matrices (one per γ)
        - gamma_candidates (np.array of γ)

    Args:
        seen_classes: binary array [total_classes], 1 = seen, 0 = unseen
        logits: [num_images, total_classes] logits (validation or test)
        gamma_min, gamma_max: search range for γ
        num_steps: number of γ candidates (dense grid)
    """

    # Convert to np arrays
    if torch.is_tensor(seen_classes):
        seen_classes = seen_classes.cpu().numpy()
    if torch.is_tensor(logits):
        logits = logits.cpu().numpy()

    seen_classes = np.array(seen_classes, dtype=bool)
    logits = np.array(logits)

    num_images, total_classes = logits.shape

    # Masks
    seen_mask = seen_classes
    unseen_mask = ~seen_classes

    # Sanity check: we expect at least one unseen class
    if unseen_mask.sum() == 0:
        print("⚠️ No unseen classes; calibration is trivial.")
        # Still return one candidate with γ = 0
        return [logits.copy()], np.array([0.0], dtype=float)

    # Dense γ grid (global bias on unseen classes)
    gamma_candidates = np.linspace(gamma_min, gamma_max, num_steps)
    calibrated_logits_list = []

    print(f"🔧 Generating {len(gamma_candidates)} calibrated logits matrices...")
    print(f"   γ range = [{gamma_min}, {gamma_max}], steps = {num_steps}")

    for gamma in gamma_candidates:
        cal_logits = logits.copy()
        # Add the same γ to all unseen-class logits for ALL samples
        cal_logits[:, unseen_mask] += gamma
        calibrated_logits_list.append(cal_logits)

    # Verify shapes
    for i, cal in enumerate(calibrated_logits_list):
        if cal.shape != logits.shape:
            print(f"❌ ERROR: Calibrated matrix {i} shape mismatch: {cal.shape} != {logits.shape}")
            return [], gamma_candidates

    print("✅ Calibration candidates generated.")
    return calibrated_logits_list, gamma_candidates


def verify_calibration_inputs(seen_classes, logits):
    """
    Verify the inputs for calibration.
    Args:
        seen_classes: 1D array indicating seen classes (0/1 or bool)
        logits: 2D array [num_images, total_classes]
    """
    try:
        if torch.is_tensor(seen_classes):
            seen_classes = seen_classes.cpu().numpy()
        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()

        seen_classes = np.array(seen_classes)
        logits = np.array(logits)

        # Dimension checks
        if len(seen_classes.shape) != 1:
            print(f"ERROR: seen_classes should be 1D, got shape {seen_classes.shape}")
            return False

        if len(logits.shape) != 2:
            print(f"ERROR: logits should be 2D, got shape {logits.shape}")
            return False

        # Compatibility
        if seen_classes.shape[0] != logits.shape[1]:
            print(f"ERROR: seen_classes length ({seen_classes.shape[0]}) != logits classes ({logits.shape[1]})")
            return False

        # Seen/unseen counts
        n_seen = int(np.sum(seen_classes))
        n_unseen = int(len(seen_classes) - n_seen)

        if n_seen == 0:
            print("WARNING: No seen classes found.")
            return False

        if n_unseen == 0:
            print("WARNING: No unseen classes found; calibration may not be meaningful.")

        print(f"✓ Calibration inputs verified: {logits.shape[0]} images, {logits.shape[1]} classes")
        print(f"  Seen classes: {n_seen}, Unseen classes: {n_unseen}")
        return True

    except Exception as e:
        print(f"ERROR in calibration input verification: {e}")
        return False


def evaluate_calibrated_logits(calibrated_logits_list, test_labels, class_names):
    """
    Evaluate each calibrated logits matrix.
    Returns list of dicts:
        {
          'matrix_index': i,
          'accuracy': ...,
          'balanced_accuracy': ...,
          'num_samples': ...
        }
    """
    results = []

    test_labels = np.array(test_labels)
    n_classes = len(class_names)

    print(f"🔍 Evaluating {len(calibrated_logits_list)} calibrated matrices...")

    for i, calibrated_logits in enumerate(calibrated_logits_list):
        preds = np.argmax(calibrated_logits, axis=1)

        # Overall accuracy
        accuracy = float(np.mean(preds == test_labels))

        # Balanced accuracy
        acc_per_class = []
        for cls_idx in range(n_classes):
            mask = (test_labels == cls_idx)
            if mask.sum() > 0:
                class_acc = float(np.mean(preds[mask] == test_labels[mask]))
                acc_per_class.append(class_acc)

        balanced_acc = float(np.mean(acc_per_class)) if acc_per_class else 0.0

        result = {
            'matrix_index': i,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'num_samples': int(len(test_labels)),
        }
        results.append(result)

    print("✅ Evaluation done.")
    return results


def find_best_calibration(evaluation_results):
    """
    Find best calibration based on balanced accuracy.
    """
    if not evaluation_results:
        print("❌ No evaluation results provided.")
        return None

    best_result = max(evaluation_results, key=lambda x: x['balanced_accuracy'])

    print("🎯 Best calibration (cheating upper bound):")
    print(f"  Matrix index: {best_result['matrix_index']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")

    return best_result


# ------------------------------
# Example standalone usage
# ------------------------------
if __name__ == "__main__":
    # Simple toy example with 3 classes, 2 samples
    class_names = ["class0", "class1", "class2"]

    # Only class 0 is seen
    seen_classes = np.array([1, 0, 0], dtype=int)

    # Logits for 2 samples
    logits = np.array([
        [2.0, 1.5, 0.8],  # sample 1
        [1.2, 2.1, 1.0],  # sample 2
    ])

    # True labels (cheating: used for selecting γ)
    labels = np.array([0, 1])

    # Directly test calibration on this toy setup
    if verify_calibration_inputs(seen_classes, logits):
        calibrated_logits_list, gamma_candidates = simple_calibration(
            seen_classes,
            logits,
            gamma_min=-10.0,
            gamma_max=10.0,
            num_steps=41,  # fewer steps for demo
        )

        eval_results = evaluate_calibrated_logits(
            calibrated_logits_list,
            labels,
            class_names,
        )
        best = find_best_calibration(eval_results)
        if best is not None:
            best_gamma = gamma_candidates[best['matrix_index']]
            print(f"Best γ (toy example): {best_gamma:.4f}")
