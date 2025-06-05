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


def CB_loss(logits, labels, samples_per_cls, no_of_classes, loss_type, beta, gamma, device):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      logits: A float tensor of size [batch, no_of_classes].
      labels: A int tensor of size [batch].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
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
