import copy
import numpy as np
import torch
from abc import ABC, abstractmethod
from ..common import eval
import torch.nn.functional as F
import logging
from tqdm import tqdm
import os
import sklearn.covariance

"""

ood.py

Out-of-Distribution (OOD) detection module for the pipeline.
---------------------------------------------------------

This module contains classes and methods for handling OOD detection. Including an abstract base class `OODModule` and concrete implementations for different OOD methods:

    -- `OODAll`: No OOD detection, all samples are considered out-of-distribution and selected.
    -- `OODNone`: No OOD detection, all samples are considered in-distribution and not selected.
    -- `OODOracle`: OOD detection using oracle labels, where the model is evaluated on the training dataset and the incorrect samples are selected.
    -- 'OODMCM': OOD detection using maximum concept matching (MCM)
    -- 'OODPastMahalanobis': OOD detection with Mahalanobis distance-based confidence score
"""


class OODModule(ABC):
    """Abstract base class for OOD detection modules.

        Abstract methods:
            - `process`: Main function of the OOD detection module. Returns a classifier and a mask. Where the mask is a boolean array with same length as the training dataset, `1` for out-of-distribution samples (selected) and `0` for in-distribution samples (not selected).
    """
    def __init__(self, ood_config, common_config, class_names, args, device):
        self.ood_config = ood_config
        self.common_config = common_config
        self.class_names = class_names
        self.args = args
        self.device = device
        self.lambda_thresh = None

    @abstractmethod
    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        pass


class OODAll(OODModule):
    """No OOD detection, all samples are considered out-of-distribution and selected.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        return classifier, copy.deepcopy(train_mask)


class OODNone(OODModule):
    """No OOD detection, all samples are considered in-distribution and not selected.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        return classifier, np.zeros_like(train_mask)


class OODOracle(OODModule):
    """OOD detection using oracle labels, where the model is evaluated on the training dataset and the incorrect samples are selected.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        train_dset.eval()
        loader = torch.utils.data.DataLoader(train_dset, batch_size=self.common_config['eval_batch_size'], shuffle=False)
        _, preds_arr, labels_arr = eval(classifier, loader, self.device)
        train_dset.train()
        correct_mask = (preds_arr == labels_arr)
        mask_copy = copy.deepcopy(train_mask)
        mask_copy[correct_mask] = 0
        return classifier, mask_copy

class OODRandom(OODModule):
    """Random OOD detection, where a random subset of samples is selected as out-of-distribution.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        np.random.seed(self.ood_config.get('random_seed', 42))
        num_samples = len(train_dset)
        random_indices = np.random.choice(num_samples, size=int(num_samples * self.ood_config.get('random_fraction', 0.1)), replace=False)
        mask_copy = np.zeros_like(train_mask)
        mask_copy[random_indices] = 1
        return classifier, mask_copy

class OODClsRandom(OODModule):
    """Random OOD detection based on class labels, where a random subset of samples from each class is selected as out-of-distribution.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        np.random.seed(self.ood_config.get('random_seed', 42))
        num_classes = len(self.class_names)
        labels = np.array([s.label for s in train_dset.samples])
        class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
        mask_copy = np.zeros_like(train_mask)
        for i, indices in class_indices.items():
            random_indices = np.random.choice(indices, size=int(len(indices) * self.ood_config.get('random_fraction', 0.5)), replace=False)
            mask_copy[random_indices] = 1
        
        return classifier, mask_copy

class OODMCMEval(OODModule):
    """
    MCM-based OOD detection: Computes image features, compares with text features,
    calculates max softmax score (MCM). Flags samples with low MCM as OOD.
    """

    def process(self, classifier, curr_eval_dset, last_eval_dset, train_mask, is_first_ckp=False, is_zs=False, is_train=False):
        classifier.eval()
        curr_eval_loader = torch.utils.data.DataLoader(
                curr_eval_dset,
                batch_size=self.common_config['eval_batch_size'],
                shuffle=False,
            )
        if last_eval_dset is not None:
            last_eval_loader = torch.utils.data.DataLoader(
                    last_eval_dset,
                    batch_size=self.common_config['eval_batch_size'],
                    shuffle=False,
                )

        all_scores = []
        all_gap_scores = []

        # Get text embeddings (head) — normalize
        text_features = classifier.head.weight.to(self.device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Score each sample
        for batch_idx, (inputs, labels, _, _, _) in tqdm(enumerate(curr_eval_loader), total=len(curr_eval_loader)):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                # Extract image features
                image_features = classifier.visual_model(inputs)
                image_features = image_features.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Similarity -> logits
                logits = image_features @ text_features.T
                logits = logits / self.ood_config.get("temperature", 1)

                # Softmax -> MCM score
                smax = F.softmax(logits, dim=-1)
                mcm = torch.max(smax, dim=-1).values  # Shape: (batch_size,)

                all_scores.append(mcm.cpu().numpy())

                # Confidence gap = top1 - top2
                top2 = torch.topk(smax, 2, dim=-1).values
                gap = (top2[:, 0] - top2[:, 1])
                all_gap_scores.append(gap.cpu().numpy())

        if len(all_scores) == 0:
            logging.warning("[OODMCM] No MCM scores computed. Skipping OOD detection for this checkpoint.")
            return classifier, np.zeros_like(train_mask)

        # Assume: all_scores = list of np arrays, one per batch
        ckp_scores = np.concatenate(all_scores, axis=0)
        gap_scores = np.concatenate(all_gap_scores, axis=0)

        # First checkpoint only: set the threshold
        if is_first_ckp:
            self.lambda_thresh = np.percentile(ckp_scores, 5)
            logging.info(f"[OOD] λ threshold (5th percentile) set to: {self.lambda_thresh:.4f}")

        # Compute mean score for logging
        ckp_mean = np.mean(ckp_scores)
        ood_pct = np.sum(ckp_scores < self.lambda_thresh) / len(ckp_scores) * 100
        gap_mean = np.mean(gap_scores)

        logging.info(f"[OOD] Checkpoint mean: {ckp_mean:.4f}")
        logging.info(f"[OOD] {ood_pct:.2f}% samples flagged as OOD")
        logging.info(f"[OOD] Mean softmax gap (Top-1 - Top-2): {gap_mean:.4f}")

        # # Make mask
        # mask_copy = copy.deepcopy(train_mask)
        # mask_copy[:] = ckp_scores < lambda_thresh

        # return classifier, mask_copy
        return classifier, np.zeros_like(train_mask)

class OODMCM(OODModule):
    """
    MCM-based OOD detection: Computes image features, compares with text features,
    calculates max softmax score (MCM). Flags samples with low MCM as OOD.
    """

    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        classifier.eval()

        loader = torch.utils.data.DataLoader(
            eval_dset,
            batch_size=self.common_config['eval_batch_size'],
            shuffle=False,
        )

        all_scores = []
        all_gap_scores = []

        # Get text embeddings (head) — normalize
        text_features = classifier.head.weight.to(self.device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Score each sample
        for batch_idx, (inputs, labels, _, _, _) in tqdm(enumerate(loader), total=len(loader)):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                # Extract image features
                image_features = classifier.visual_model(inputs)
                image_features = image_features.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Similarity -> logits
                logits = image_features @ text_features.T
                logits = logits / self.ood_config.get("temperature", 1)

                # Softmax -> MCM score
                smax = F.softmax(logits, dim=-1)
                mcm = torch.max(smax, dim=-1).values  # Shape: (batch_size,)

                all_scores.append(mcm.cpu().numpy())

                # Confidence gap = top1 - top2
                top2 = torch.topk(smax, 2, dim=-1).values
                gap = (top2[:, 0] - top2[:, 1])
                all_gap_scores.append(gap.cpu().numpy())

        if len(all_scores) == 0:
            logging.warning("[OODMCM] No MCM scores computed. Skipping OOD detection for this checkpoint.")
            return classifier, np.zeros_like(train_mask)

        # Assume: all_scores = list of np arrays, one per batch
        ckp_scores = np.concatenate(all_scores, axis=0)
        gap_scores = np.concatenate(all_gap_scores, axis=0)

        # First checkpoint only: set the threshold
        if is_first_ckp:
            self.lambda_thresh = np.percentile(ckp_scores, 5)
            logging.info(f"[OOD] λ threshold (5th percentile) set to: {self.lambda_thresh:.4f}")

        # Compute mean score for logging
        ckp_mean = np.mean(ckp_scores)
        ood_pct = np.sum(ckp_scores < self.lambda_thresh) / len(ckp_scores) * 100
        gap_mean = np.mean(gap_scores)

        logging.info(f"[OOD] Checkpoint mean: {ckp_mean:.4f}")
        logging.info(f"[OOD] {ood_pct:.2f}% samples flagged as OOD")
        logging.info(f"[OOD] Mean softmax gap (Top-1 - Top-2): {gap_mean:.4f}")

        # # Make mask
        # mask_copy = copy.deepcopy(train_mask)
        # mask_copy[:] = ckp_scores < lambda_thresh

        # return classifier, mask_copy
        return classifier, np.zeros_like(train_mask)

class OODPastMahalanobis(OODModule):
    """
    Mahalanobis-based OOD detection with Past Checkpoints:
    - Incrementally fits mean & cov with past train data
    - Uses future test data to compute Mahalanobis scores and updates lambda_thresh dynamically
    """

    def __init__(self, ood_config, common_config, class_names, args, device):
        super().__init__(ood_config, common_config, class_names, args, device)
        self.mean_cov_file = ood_config.get('mean_cov_file', './mean_cov.npz')
        self.class_means = None
        self.cov_inv = None
        self.lambda_thresh = None

    def process(self, classifier, train_dset, eval_dset, train_mask, is_first_ckp=False, is_zs=False):
        
        if is_first_ckp:
            logging.info(f"[OOD] Checkpoint mean: {0.0}")
            logging.info(f"[OOD] {0.0}% samples flagged as OOD")
            return classifier, np.zeros_like(train_mask)
        
        classifier.eval()

        train_dset.eval()
        loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=self.common_config['eval_batch_size'],
            shuffle=False,
        )

        new_features = [[] for _ in range(len(self.class_names))]

        for inputs, labels, _, _, _ in tqdm(loader, desc="[PastMahalanobis] Extract train features"):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                feat = classifier.visual_model(inputs) 
                feat = feat.float()
                feat = F.normalize(feat, dim=-1)
                feat = feat.cpu().numpy()

            for f, y in zip(feat, labels):
                new_features[y].append(f)

        train_dset.train()

        # if zs then should load all the features
        # if accumu then should train all the past which should be the same as train_dset
        if is_zs:
            if os.path.exists(self.mean_cov_file):
                saved = np.load(self.mean_cov_file, allow_pickle=True)
                old_features = saved['all_features'].tolist()
                logging.info(f"[PastMahalanobis] Loaded previous features from {self.mean_cov_file}")
            else:
                old_features = [[] for _ in range(len(self.class_names))]
                logging.info(f"[PastMahalanobis] No previous features found, starting fresh.")

            all_features = []
            for old, new in zip(old_features, new_features):
                all_feats = old + new
                all_features.append(all_feats)
        else:
            all_features = new_features

        # === new mean & cov ===
        class_means = []
        valid_features = []  # keep only non-empty classes
        for feats in all_features:
            if len(feats) == 0:
                logging.warning("[PastMahalanobis] Skipping empty class with no samples.")
                continue
            feats = np.stack(feats)
            mu = np.mean(feats, axis=0)
            class_means.append(mu)
            valid_features.append(feats)

        if len(valid_features) == 0:
            raise ValueError("[PastMahalanobis] All classes are empty! Cannot fit mean/cov.")

        X = []
        for feats, mu in zip(valid_features, class_means):
            X.append(feats - mu)
        X = np.vstack(X)

        cov = sklearn.covariance.EmpiricalCovariance(assume_centered=False).fit(X)
        cov_inv = cov.precision_

        # update the mean and cov
        self.class_means = np.stack(class_means)
        self.cov_inv = cov_inv

        # Past Mahalanobis scores to determine the new threshold for ood
        past_scores = []
        for class_id, feats in enumerate(all_features):
            for f in feats:
                dists = []
                for mu in class_means:
                    diff = f - mu
                    dist = - 0.5 * diff.T @ cov_inv @ diff 
                    dists.append(dist)
                # score = -min(dists)
                score = max(dists)
                past_scores.append(score)
        past_scores = np.array(past_scores)

        self.lambda_thresh = np.percentile(past_scores, 5)
        logging.info(f"[OOD] λ threshold (5th percentile) set to: {self.lambda_thresh:.4f}")

        # future 
        eval_loader = torch.utils.data.DataLoader(
            eval_dset,
            batch_size=self.common_config['eval_batch_size'],
            shuffle=False,
        )

        future_scores = []

        for inputs, _, _, _, _ in tqdm(eval_loader, desc="[PastMahalanobis] Calculate Mahalanobis scores"):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                feat = classifier.visual_model(inputs)
                feat = feat.float()
                feat = F.normalize(feat, dim=-1)
                feat = feat.cpu().numpy()

            for f in feat:
                dists = []
                for mu in self.class_means:
                    diff = f - mu
                    dist = - 0.5 * diff.T @ self.cov_inv @ diff
                    dists.append(dist)
                # score = -min(dists)
                score = max(dists)
                future_scores.append(score)

        future_scores = np.array(future_scores)
        mean_score = np.mean(future_scores)
        ood_pct = np.sum(future_scores < self.lambda_thresh) / len(future_scores) * 100

        logging.info(f"[OOD] Checkpoint mean: {mean_score:.4f}")
        logging.info(f"[OOD] {ood_pct:.2f}% samples flagged as OOD")

        # save new features, mean, cov_inv, lambda_thresh
        np.savez(self.mean_cov_file,
                 all_features=np.array(all_features, dtype=object),
                 class_means=self.class_means,
                 cov_inv=self.cov_inv,
                 lambda_thresh=self.lambda_thresh)

        logging.info(f"[PastMahalanobis] Saved updated mean & cov & λ to {self.mean_cov_file}")

        # # === OOD mask ===
        # mask_copy = copy.deepcopy(train_mask)
        # mask_copy[:] = future_scores < self.lambda_thresh

        # return classifier, mask_copy
        return classifier, np.zeros_like(train_mask)


OOD_METHODS = {
    'all': OODAll,
    'none': OODNone,
    'oracle': OODOracle,
    'random': OODRandom,
    'cls_random': OODClsRandom,
    'mcm': OODMCM,
    'past_mahalanobis': OODPastMahalanobis,
}


def get_ood_module(ood_config, common_config, class_names, args, device):
    """Entry point for OOD detection module.

    Args:
        ood_config (dict): OOD configuration dictionary. From the config file.
        common_config (dict): Common configuration dictionary. From the config file.
        class_names (list): List of class names. From the config file.
        device (torch.device): Device to use for computation.
    Returns:
        OODModule: Instance of the OODModule.

    """
    method = ood_config.get('method', 'none')
    if method not in OOD_METHODS:
        raise ValueError(f"Unknown OOD method: {method}.")
    return OOD_METHODS[method](ood_config, common_config, class_names, args, device)
