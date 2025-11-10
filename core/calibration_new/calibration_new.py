import copy
import json
import torch
from torch.nn import functional as F
import core.calibration_new.common as data_api
import numpy as np
from . import math

class Extraction:

    def __init__(self, features, logits, labels, data_ind=None):
        self.features = features
        self.logits = logits
        self.labels = labels
        if data_ind is not None:
            self.data_ind = data_ind
        else:
            self.data_ind = None


def run_calibration(dataset, eval_dataset, training_loader, eval_loader, model, seen_classes, unseen_classes, eval_seen_classes, eval_unseen_classes, device):
    with torch.no_grad():
      model.eval()
      all_classes = torch.tensor([dataset.class_name_idx[name] for name in dataset.class_names], dtype=torch.long)
      visible_classes = torch.tensor([dataset.class_name_idx[name] for name in seen_classes], dtype=torch.long) if seen_classes else torch.tensor([], dtype=torch.long)
      invisible_classes = torch.tensor([dataset.class_name_idx[name] for name in unseen_classes], dtype=torch.long) if unseen_classes else torch.tensor([], dtype=torch.long)
      eval_visible_classes = torch.tensor([eval_dataset.class_name_idx[name] for name in eval_seen_classes], dtype=torch.long) if eval_seen_classes else torch.tensor([], dtype=torch.long)
      eval_invisible_classes = torch.tensor([eval_dataset.class_name_idx[name] for name in eval_unseen_classes], dtype=torch.long) if eval_unseen_classes else torch.tensor([], dtype=torch.long)
      num_classes = len(all_classes)

      domain_info = data_api.DomainInfo(all_classes,
                                        visible_classes,
                                        invisible_classes,
                                        num_classes)
      
      eval_domain_info = data_api.DomainInfo(all_classes,
                                            eval_visible_classes,
                                            eval_invisible_classes,
                                            num_classes)
      
      oracle_training_extraction  = extract(training_loader, model, device)
      oracle_eval_extraction = extract(eval_loader, model, device)
      
      # Baseline Calibration (mean, max), which is in real-world use case
      baseline_calib_factor_mean, baseline_calib_factor_max = evaluate_baseline_calibration(domain_info, oracle_training_extraction)
      
      # Oracle Calibration, which is an upper bound
      upper_bound_balanced_acc = _get_curve_results(eval_domain_info, oracle_eval_extraction)
      
      return baseline_calib_factor_mean, baseline_calib_factor_max, upper_bound_balanced_acc

def _get_curve_results(domain_info, extraction):
    curve_results=[]

    # Compute seen and unseen sample masks
    visible_class_mask = torch.tensor([l.item() in domain_info.visible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)
    invisible_class_mask = torch.tensor([l.item() in domain_info.invisible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)

    # Increase unseen accuracy
    logits_copy = extraction.logits.clone().to(torch.float64)
    final = False
    accumulate_shifting = 0.
    while not final:
        unseen_shifting, final = _compute_shifting(
                logits_copy, domain_info.visible_classes, domain_info.invisible_classes,
                mode='positive')
        logits_copy[:, domain_info.invisible_classes] += unseen_shifting
        accumulate_shifting += unseen_shifting
        balanced_acc, visible_balanced_acc, invisible_balanced_acc = _compute_accuracy(logits_copy, extraction.labels, visible_class_mask, invisible_class_mask)
        curve_results.append([balanced_acc, visible_balanced_acc, invisible_balanced_acc,        accumulate_shifting])

    # Increase seen accuracy
    logits_copy = extraction.logits.clone().to(torch.float64)
    final = False
    accumulate_shifting = 0.
    while not final:
        unseen_shifting, final = _compute_shifting(
                logits_copy, domain_info.visible_classes, domain_info.invisible_classes,
                mode='negative')
        logits_copy[:, domain_info.invisible_classes] -= unseen_shifting
        accumulate_shifting -= unseen_shifting
        balanced_acc, visible_balanced_acc, invisible_balanced_acc = _compute_accuracy(logits_copy, extraction.labels, visible_class_mask, invisible_class_mask)
        curve_results.append([balanced_acc, visible_balanced_acc, invisible_balanced_acc, accumulate_shifting])

    # Get trade-off curve
    curve_results = torch.tensor(curve_results)
    curve_results = curve_results[torch.argsort(curve_results[:, 1])]  # Sort by seen acc.
    trade_off_curve = curve_results[:, 1:3]

    return curve_results[trade_off_curve.mean(dim=1).argmax().item()][0]

def _compute_accuracy(all_logits, all_labels, visible_class_mask=None, invisible_class_mask=None):
    pred_false = []
    pred_true = []
    confidences = F.softmax(all_logits, dim=1)
    preds = all_logits.argmax(dim=1)

    for i in range(len(all_labels)):
        if preds[i] == all_labels[i]:
            pred_true.append((all_labels[i].item(), preds[i].item(), confidences[i].max().item()))
        else:
            pred_false.append((all_labels[i].item(), preds[i].item(), confidences[i].max().item()))

    preds_arr = preds.cpu().numpy()
    labels_arr = all_labels.cpu().numpy()
    
    acc_per_class = []
    for i in range(len(all_labels)):
        mask = labels_arr == i
        if mask.sum() == 0:
            continue
        acc_per_class.append((preds_arr[mask] == labels_arr[mask]).mean())
    acc_per_class = np.array(acc_per_class)
    balanced_acc = acc_per_class.mean()
    
    # Compute visible_balanced_acc
    if visible_class_mask is not None:
        visible_preds_arr = preds_arr[visible_class_mask.cpu().numpy()]
        visible_labels_arr = labels_arr[visible_class_mask.cpu().numpy()]
        visible_acc_per_class = []
        for i in range(len(all_labels)):
            mask = visible_labels_arr == i
            if mask.sum() == 0:
                continue
            visible_acc_per_class.append((visible_preds_arr[mask] == visible_labels_arr[mask]).mean())
        visible_acc_per_class = np.array(visible_acc_per_class)
        visible_balanced_acc = visible_acc_per_class.mean() if len(visible_acc_per_class) > 0 else 0
    else:
        visible_balanced_acc = balanced_acc
    
    # Compute invisible_balanced_acc
    if invisible_class_mask is not None:
        invisible_preds_arr = preds_arr[invisible_class_mask.cpu().numpy()]
        invisible_labels_arr = labels_arr[invisible_class_mask.cpu().numpy()]
        invisible_acc_per_class = []
        for i in range(len(all_labels)):
            mask = invisible_labels_arr == i
            if mask.sum() == 0:
                continue
            invisible_acc_per_class.append((invisible_preds_arr[mask] == invisible_labels_arr[mask]).mean())
        invisible_acc_per_class = np.array(invisible_acc_per_class)
        invisible_balanced_acc = invisible_acc_per_class.mean() if len(invisible_acc_per_class) > 0 else 0
    else:
        invisible_balanced_acc = balanced_acc
    
    return balanced_acc, visible_balanced_acc, invisible_balanced_acc

def _compute_shifting(all_logits, visible_classes, invisible_classes, mode='positive'):
    assert mode in ['positive', 'negative']

    # Compute maximum logits for seen and unseen classes
    max_seen_logits = all_logits[:, visible_classes].max(dim=1)[0]
    max_unseen_logits = all_logits[:, invisible_classes].max(dim=1)[0]

    # Determine valid indices based on mode
    if mode == 'positive':
        valid = max_seen_logits >= max_unseen_logits
        diff = (max_seen_logits[valid] - max_unseen_logits[valid]).sort()[0]
    else:
        valid = max_seen_logits <= max_unseen_logits
        diff = (max_unseen_logits[valid] - max_seen_logits[valid]).sort()[0]

    # Check for invalid differences
    assert (diff < 0).sum() == 0

    # Handle different cases based on diff shape
    if diff.shape[0] == 0:
        return 0., True
    elif diff.shape[0] == 1:
        return diff[0] + 1., True
    else:
        first = diff[0]
        for d in diff[1:]:
            if d != first:
                second = d
                break
            else:
                second = d
        if first != second:
            return (first + second) / 2., False
        else:
            return first + 1., True

def evaluate_baseline_calibration(domain_info, extraction):
    wrong_visible_logits = []
    invisible_logits = []

    for idx in range(extraction.logits.shape[0]):
        logit = extraction.logits[idx]
        label = extraction.labels[idx]

        invisible_logits.append(logit[domain_info.invisible_classes].mean().item())

        wrong_visible_classes = [x.item() for x in domain_info.visible_classes if x.item() != label.item()]
        wrong_visible_logits.append(logit[wrong_visible_classes].mean().item())

    # Original: Mean based
    baseline_calib_factor_mean = torch.tensor(wrong_visible_logits).mean().item() - torch.tensor(invisible_logits).mean().item()

    # 2nd option: Max based
    baseline_calib_factor_max = torch.tensor(wrong_visible_logits).max().item() - torch.tensor(invisible_logits).max().item()

    # calib_all_logits = extraction.logits.clone()
    # calib_all_logits[:, domain_info.invisible_classes] += baseline_calib_factor

    # Compute seen and unseen sample masks    

    # generate_accu_metric(domain_info, score, label, data_ind)
    # metric = get_accuracies(domain_info, calib_all_logits, extraction.labels, extraction.data_ind)

    return  baseline_calib_factor_mean, baseline_calib_factor_max

def mask_score(score, row_mask, column_mask):
    score = copy.deepcopy(score)
    score[~row_mask] = -torch.inf
    score[:, ~column_mask] = -torch.inf
    return score

def mask_predict_score(score, label, row_mask=None, column_mask=None):
    if row_mask is None:
        row_mask = torch.ones(score.shape[0], dtype=torch.bool)
    if column_mask is None:
        column_mask = torch.ones(score.shape[1], dtype=torch.bool)
    masked_score = mask_score(score, row_mask, column_mask)
    masked_score = masked_score[row_mask]
    masked_label = label[row_mask]
    accuracy = math.topk_accuracy(masked_score, masked_label)
    return accuracy

def get_accuracies(domain_info, score, label, data_ind):
    # Unpack domain_info
    visible_classes = domain_info.visible_classes.type(torch.int64)
    invisible_classes = domain_info.invisible_classes.type(torch.int64)
    remaining_classes = domain_info.remaining_classes.type(torch.int64)
    dim_score = score.shape[1]

    # Masks
    visible_row_mask = torch.isin(data_ind, domain_info.visible_ind)
    invisible_row_mask = torch.isin(data_ind, domain_info.invisible_ind)
    remaining_row_mask = torch.isin(data_ind, domain_info.remaining_ind)
    visible_column_mask = torch.zeros(dim_score, dtype=torch.bool)
    visible_column_mask[visible_classes] = 1
    invisible_column_mask = torch.zeros(dim_score, dtype=torch.bool)
    invisible_column_mask[invisible_classes] = 1
    remaining_column_mask = torch.zeros(dim_score, dtype=torch.bool)
    remaining_column_mask[remaining_classes] = 1
    
    # Calculate metric
    #  From all classes / Over all classes
    all_all_accuracy = mask_predict_score(score, label, row_mask=None, column_mask=None)
    
    #  From visible classes / Over all classes
    visible_all_accuracy = mask_predict_score(score, label, row_mask=visible_row_mask, column_mask=None)
    
    #  From invisible classes / Over all classes
    invisible_all_accuracy = mask_predict_score(score, label, row_mask=invisible_row_mask, column_mask=None)

    #  From remaining classes / Over all classes
    remaining_all_accuracy = mask_predict_score(score, label, row_mask=remaining_row_mask, column_mask=None)
    
    #  From visible classes / Over visible classes
    visible_visible_accuracy = mask_predict_score(score, label, row_mask=visible_row_mask, column_mask=visible_column_mask)
    
    #  From invisible classes / Over invisible classes
    invisible_invisible_accuracy = mask_predict_score(score, label, row_mask=invisible_row_mask, column_mask=invisible_column_mask)

    #  From remaining classes / Over remaining classes
    remaining_remaining_accuracy = mask_predict_score(score, label, row_mask=remaining_row_mask, column_mask=remaining_column_mask)
    
    # Package accuracies
    accuracies = {
        'All/All Accuracy': all_all_accuracy,
        'Visible/All Accuracy': visible_all_accuracy,
        'Invisible/All Accuracy': invisible_all_accuracy,
        'Remaining/All Accuracy': remaining_all_accuracy,
        'Visible/Visible Accuracy': visible_visible_accuracy,
        'Invisible/Invisible Accuracy': invisible_invisible_accuracy,
        'Remaining/Remaining Accuracy': remaining_remaining_accuracy
    }
    return accuracies

def extract(loader, model, device):

      # Initialize extraction
      features = []
      logits = []
      labels = []
      # data_ind = []

      # Extract features
      # for _data_ind, (_X, _y) in loader:
      for _data_ind, (_X, _y, _, _, _) in enumerate(loader):
          _logits, _features, _labels = extract_batch(_X, _y, model, device)
          features.append(_features)
          logits.append(_logits)
          labels.append(_labels)
          # data_ind.append(_data_ind)
      
      # Concatenate features
      features = torch.cat(features, dim=0)
      logits = torch.cat(logits, dim=0)
      labels = torch.cat(labels, dim=0)
      # data_ind = torch.cat(data_ind, dim=0).to(device)

      # Ensemble extraction
      extraction = Extraction(features, logits, labels)
      return extraction
    
def extract_batch(X, y, model, device):
    # Set model
    if model is None:
        model = model

    # Extract batch
    X = X.to(device)
    y = y.to(device)
    logits, features = model(X, return_feats=True)

    # Postprocess
    return logits, features, y

def extract_seen_unseen(is_train, common_config, label_type, checkpoint_analysis):
    # Analyze checkpoint classes for seen/unseen
    if is_train:
        config_path = common_config["train_data_config_path"]
        tag = 'train'
    else:
        config_path = common_config["eval_data_config_path"]
        tag = 'eval'
    
    # Ensure checkpoint_analysis is a dict and that the tag key exists
    if checkpoint_analysis is None:
        checkpoint_analysis = {}
    if tag not in checkpoint_analysis:
        checkpoint_analysis[tag] = {}
    
    with open(config_path, 'r') as f:
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
    
    for checkpoint in checkpoints:
        checkpoint_classes = set()
        for entry in data[checkpoint]:
            checkpoint_classes.add(entry[label_type])
        cumulative_seen.update(checkpoint_classes)
        seen = sorted(list(cumulative_seen))
        unseen = sorted(list(set(all_classes) - cumulative_seen))
        checkpoint_analysis[tag][checkpoint] = {'seen': seen, 'unseen': unseen}
    
    return checkpoint_analysis