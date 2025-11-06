import copy
import torch
import core.calibration_new.common as data_api

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


def run_alg_calibration(dataset, training_loader, model, seen_classes, unseen_classes, device):
    with torch.no_grad():
      model.eval()
      all_classes = torch.tensor([dataset.class_name_idx[name] for name in dataset.class_names], dtype=torch.long)
      visible_classes = torch.tensor([dataset.class_name_idx[name] for name in seen_classes], dtype=torch.long) if seen_classes else torch.tensor([], dtype=torch.long)
      invisible_classes = torch.tensor([dataset.class_name_idx[name] for name in unseen_classes], dtype=torch.long) if unseen_classes else torch.tensor([], dtype=torch.long)
      num_classes = len(all_classes)

      domain_info = data_api.DomainInfo(all_classes,
                                        visible_classes,
                                        invisible_classes,
                                        num_classes)
      oracle_training_extraction  = extract(training_loader, model, device)
      baseline_calib_factor = evaluate_baseline_calibration(domain_info, oracle_training_extraction)
      return baseline_calib_factor
    
def evaluate_baseline_calibration(domain_info, extraction):
    wrong_visible_logits = []
    invisible_logits = []

    for idx in range(extraction.logits.shape[0]):
        logit = extraction.logits[idx]
        label = extraction.labels[idx]

        invisible_logits.append(logit[domain_info.invisible_classes].mean().item())

        wrong_visible_classes = [x.item() for x in domain_info.visible_classes if x.item() != label.item()]
        wrong_visible_logits.append(logit[wrong_visible_classes].mean().item())

    baseline_calib_factor= torch.tensor(wrong_visible_logits).mean().item() - torch.tensor(invisible_logits).mean().item()

    calib_all_logits = extraction.logits.clone()
    calib_all_logits[:, domain_info.invisible_classes] += baseline_calib_factor

    # Compute seen and unseen sample masks    

    # generate_accu_metric(domain_info, score, label, data_ind)
    # metric = get_accuracies(domain_info, calib_all_logits, extraction.labels, extraction.data_ind)

    return baseline_calib_factor

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