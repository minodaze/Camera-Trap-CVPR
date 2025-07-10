import numpy as np
import torch
import torch.nn.functional as F

def loss_fn_kd(scores, target_scores, T=2.):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    kd_loss = (-1 * targets_norm * log_scores_norm).sum(dim=1).mean() * T ** 2
    return kd_loss

def SupConLoss(features, device, labels=None, mask=None, target_labels=None, reduction='mean', temperature=0.5, base_temperature=0.07, contrast_mode='all'):
    assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

    if len(features.shape) < 3:
      raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
    if len(features.shape) > 3:
      features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
      raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
      mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
      labels = labels.contiguous().view(-1, 1)
      if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
      mask = torch.eq(labels, labels.T).float().to(device)
    else:
      mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
      anchor_feature = features[:, 0]
      anchor_count = 1
    elif contrast_mode == 'all':
      anchor_feature = contrast_feature
      anchor_count = contrast_count
    else:
      raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
      torch.matmul(anchor_feature, contrast_feature.T),
      temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
      torch.ones_like(mask),
      1,
      torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
      0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos

    curr_class_mask = torch.zeros_like(labels)
    for tc in target_labels:
      curr_class_mask += (labels == tc)
    curr_class_mask = curr_class_mask.view(-1).to(device)
    loss = curr_class_mask * loss.view(anchor_count, batch_size)

    if reduction == 'mean':
      loss = loss.mean()
    elif reduction == 'none':
      loss = loss.mean(0)
    else:
      raise ValueError('loss reduction not supported: {}'.
                             format(reduction))
    return loss

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(logits, labels, samples_per_cls, no_of_classes, loss_type, beta, gamma, device, use_per_class_beta=False):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      logits: A float tensor of size [batch, no_of_classes].
      labels: A int tensor of size [batch].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss. If use_per_class_beta=True, this is ignored.
      gamma: float. Hyperparameter for Focal loss.
      device: device to place tensors on.
      use_per_class_beta: bool. If True, use β_i = (n_i - 1)/n_i for each class. If False, use global beta.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    samples_per_cls = np.array(samples_per_cls)
    
    if use_per_class_beta:
        # Per-class beta: β_i = (n_i - 1)/n_i
        beta_per_class = (samples_per_cls - 1) / samples_per_cls
        # Handle case where n_i = 1 (would cause division by zero)
        beta_per_class[samples_per_cls == 1] = 0.0
        effective_num = 1.0 - np.power(beta_per_class, samples_per_cls)
        weights = (1.0 - beta_per_class) / effective_num
    else:
        # Global beta (original implementation)
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
    
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        assert False, "Not implemented"
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    
    return cb_loss

# def CDT_loss(logits, labels, samples_per_cls, no_of_classes, gamma=1.0, device='cuda'):
#     """Compute the Class-Dependent Temperatures (CDT) Loss between `logits` and the ground truth `labels`.
    
#     CDT Loss is designed to compensate for feature deviation in class-imbalanced datasets
#     by applying class-dependent temperatures that are inversely proportional to the number
#     of training instances.
    
#     Reference: Class-Dependent Temperatures (CDT) approach for learning ConvNet classifier
#     with class-imbalanced data.
    
#     Args:
#         logits: A float tensor of size [batch, no_of_classes].
#         labels: A int tensor of size [batch].
#         samples_per_cls: A python list of size [no_of_classes] containing number of samples per class.
#         no_of_classes: total number of classes. int
#         gamma: float. Hyperparameter controlling the temperature scaling (gamma >= 0).
#         device: device to place tensors on.
        
#     Returns:
#         cdt_loss: A float tensor representing CDT loss
#     """
#     batch_size = logits.size(0)
    
#     # Convert samples_per_cls to tensor
#     samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32, device=device)
    
#     # Calculate class-dependent temperatures
#     # a_c = (N_max / N_c)^gamma, where N_max is the largest number of training instances
#     N_max = torch.max(samples_per_cls)
#     temperatures = torch.pow(N_max / samples_per_cls, gamma)
    
#     # For the major class (most samples), a_c = 1
#     # For other classes, a_c > 1 if gamma > 0
#     # When gamma = 0, we recover the conventional ERM objective
    
#     # Apply class-dependent temperatures to logits
#     # For each sample, use the temperature corresponding to its true class
#     sample_temperatures = temperatures[labels]  # [batch_size]
    
#     # Scale logits by class-dependent temperatures
#     # w_y^T f_θ(x_n) / a_yn
#     scaled_logits = logits / sample_temperatures.unsqueeze(1)
    
#     # Compute CDT loss using the scaled logits
#     cdt_loss = F.cross_entropy(scaled_logits, labels)
    
#     return cdt_loss


def CDT_loss(logits, labels, samples_per_cls, no_of_classes, gamma=0.3, device='cuda'):
    """Manual implementation of CDT Loss following the exact formula from the paper.
    
    This implementation follows equation (5) from the CDT paper:
    -∑_n log[ exp(w_yn^T f_θ(x_n) / a_yn) / ∑_c exp(w_c^T f_θ(x_n) / a_c) ]
    
    Args:
        logits: A float tensor of size [batch, no_of_classes].
        labels: A int tensor of size [batch].
        samples_per_cls: A python list of size [no_of_classes] containing number of samples per class.
        no_of_classes: total number of classes. int
        gamma: float. Hyperparameter controlling the temperature scaling (gamma >= 0).
        device: device to place tensors on.
        
    Returns:
        cdt_loss: A float tensor representing CDT loss
    """
    batch_size = logits.size(0)
    
    # Convert samples_per_cls to tensor
    samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32, device=device)
    

    # Calculate class-dependent temperatures
    N_max = torch.max(samples_per_cls)
    temperatures = torch.pow(N_max / samples_per_cls, gamma)  # [no_of_classes]
    
    # Scale logits by temperatures for all classes
    # logits: [batch_size, no_of_classes]
    # temperatures: [no_of_classes]
    scaled_logits = logits / temperatures.unsqueeze(0)  # [batch_size, no_of_classes]
    
    # Compute log probabilities manually following equation (5)
    # For numerical stability, subtract max
    logits_max, _ = torch.max(scaled_logits, dim=1, keepdim=True)
    scaled_logits_stable = scaled_logits - logits_max
    
    # Compute denominator: ∑_c exp(w_c^T f_θ(x_n) / a_c)
    exp_scaled_logits = torch.exp(scaled_logits_stable)  # [batch_size, no_of_classes]
    denominator = torch.sum(exp_scaled_logits, dim=1, keepdim=True)  # [batch_size, 1]
    
    # Compute numerator: exp(w_yn^T f_θ(x_n) / a_yn)
    numerator = exp_scaled_logits.gather(1, labels.unsqueeze(1))  # [batch_size, 1]
    
    # Compute log probabilities
    log_probs = torch.log(numerator / denominator)  # [batch_size, 1]
    
    # CDT loss is the negative log likelihood
    cdt_loss = -torch.mean(log_probs)
    
    return cdt_loss

def standard_focal_loss(logits, labels, alpha=1.0, gamma=2.0):
    """Compute the standard focal loss for multi-class classification.

    Standard Focal Loss: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A int tensor of size [batch].
      alpha: A float scalar specifying the weighting factor for rare class (default: 1.0).
      gamma: A float scalar modulating loss from hard and easy examples (default: 2.0).

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    # Compute cross entropy
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
    # Compute pt (probability of true class)
    pt = torch.exp(-ce_loss)
    
    # Compute focal loss
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()

def LDAM_loss(logits, labels, samples_per_cls, no_of_classes, C=0.5, device='cuda'):
    """Compute the Label-Distribution-Aware Margin (LDAM) Loss.

    LDAM Loss: L_LDAM((x,y);f) = -log(e^(z_y - Δ_y) / (e^(z_y - Δ_y) + Σ_{j≠y} e^z_j))
    where Δ_j = C / n_j^(1/4) for j ∈ {1,...,k}

    Args:
      logits: A float tensor of size [batch, no_of_classes].
      labels: A int tensor of size [batch].
      samples_per_cls: A python list of size [no_of_classes] containing number of samples per class.
      no_of_classes: total number of classes. int
      C: float. Hyperparameter controlling the margin scaling (default: 0.5).
      device: device to place tensors on.

    Returns:
      ldam_loss: A float tensor representing LDAM loss
    """
    batch_size = logits.size(0)
    
    # Convert samples_per_cls to tensor
    samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32, device=device)
    
    # Calculate margins: Δ_j = C / n_j^(1/4)
    margins = C / torch.pow(samples_per_cls, 1/4)  # [no_of_classes]
    
    # Apply margins to the true class logits
    # For each sample, subtract the margin corresponding to its true class
    sample_margins = margins[labels]  # [batch_size]
    
    # Create adjusted logits: z_y - Δ_y for true class, z_j for other classes
    adjusted_logits = logits.clone()
    adjusted_logits[range(batch_size), labels] -= sample_margins
    
    # Compute LDAM loss using adjusted logits
    ldam_loss = F.cross_entropy(adjusted_logits, labels)
    
    return ldam_loss

def balanced_softmax_loss(logits, labels, samples_per_cls, no_of_classes, device='cuda'):
    """Compute the Balanced Softmax Loss.

    Balanced Softmax Loss accommodates label distribution shifts between training and test sets
    by reweighting the softmax probabilities based on class frequencies.
    
    Formula: l(θ) = -log(φ_y) = -log(n_y * e^η_y / Σ_i n_i * e^η_i)
    where n_y is the number of samples for true class y, and η is the logit.

    Args:
      logits: A float tensor of size [batch, no_of_classes].
      labels: A int tensor of size [batch].
      samples_per_cls: A python list of size [no_of_classes] containing number of samples per class.
      no_of_classes: total number of classes. int
      device: device to place tensors on.

    Returns:
      bsl_loss: A float tensor representing balanced softmax loss
    """
    batch_size = logits.size(0)
    
    # Convert samples_per_cls to tensor
    samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32, device=device)
    
    # Compute reweighted logits: n_i * e^η_i for all classes
    # logits: [batch_size, no_of_classes]
    # samples_per_cls: [no_of_classes]
    reweighted_logits = logits + torch.log(samples_per_cls.unsqueeze(0))  # [batch_size, no_of_classes]
    
    # Compute balanced softmax loss using the reweighted logits
    bsl_loss = F.cross_entropy(reweighted_logits, labels)
    
    return bsl_loss
