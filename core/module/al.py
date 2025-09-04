import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import copy
import math
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import gc
from ..common import *
import pickle

"""

al.py

Active learning module for the pipeline.
---------------------------------------------------------

This module defines the abstract base class `ALModule` and its concrete implementations for different active learning strategies:

    - `ALNone`: A naive baseline, no samples are selected for training.
    - `ALAll`: A naive baseline, all samples are selected for training.
    - `ALRandom`: Randomly selects a percentage of samples for training.
    - `ALActiveFT`: Active learning using feature-based sampling. Reference: https://arxiv.org/abs/2303.14382
    - `ALMLS`: Select samples with minimum "maximum logit score".
    - `ALMSP`: Select samples with minimum "maximum softmax probability".

"""


def extract_logits_features(classifier, dset, device, batch_size=64):
    """Utility function to extract logits and features from the classifier.
    Memory-optimized version with smaller batch size and immediate GPU cleanup.
    """
    classifier.eval()
    with torch.no_grad():
        # Use smaller batch size to reduce memory usage
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4)
        features, logits = [], []
        
        for inputs, _, _, _ in loader:
            inputs = inputs.to(device)
            _logits, feature = classifier.forward(inputs, return_feats=True)
            
            # Move to CPU immediately and delete GPU tensors
            logits.append(_logits.cpu())
            features.append(feature.cpu())
            
            # Free GPU memory immediately
            del inputs, _logits, feature
            
        if len(logits) > 0:
            # Concatenate on CPU first, then move to device if needed
            logits_cpu = torch.cat(logits, dim=0)
            features_cpu = torch.cat(features, dim=0)
            
            # Clear the list to free memory
            del logits, features
            
            # Move back to device (some algorithms need GPU tensors)
            logits = logits_cpu.to(device)
            features = features_cpu.to(device)
            
            # Clean up CPU versions
            del logits_cpu, features_cpu
        else:
            # Sanity Check
            assert len(dset) == 0
            return [], []
            
    return logits, features

class ALModule(ABC):
    """Abstract base class for active learning modules.

    Abstract methods:
        - `process`: Main function of active learning module. Returns a classifier and a mask. Where the mask is a boolean array of the same length as the training dataset, `1` means the sample is selected for training, `0` means the sample is not selected.

    """
    def __init__(self, al_config, common_config, class_names, args, device):
        self.al_config = al_config
        self.common_config = common_config
        self.class_names = class_names
        self.args = args
        self.device = device

    @abstractmethod
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        pass


class ALNone(ALModule):
    """Naive baseline, no samples are selected for training.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        return classifier, np.zeros_like(train_mask)


class ALAll(ALModule):
    """Naive baseline, all samples are selected for training.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        return classifier, np.ones_like(train_mask)


class ALRandom(ALModule):
    """Randomly selects a percentage of samples for training.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        one_indices = np.where(train_mask == 1)[0]
        percentage = self.al_config.get('percentage', 0.0)
        if not (0.0 <= percentage <= 1.0):
            raise ValueError(f"Invalid percentage {percentage}, must be between 0 and 1.")
        
        n_samples = int(len(one_indices) * percentage)
        shuffled_indices = np.random.permutation(one_indices)[:n_samples]
        
        train_mask = np.zeros(train_mask.shape, dtype=train_mask.dtype)
        train_mask[shuffled_indices] = 1
        return classifier, train_mask

class ALClsRandom(ALModule):
    """Random AL selection based on class labels, where a random subset of samples from each class is selected for training.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        np.random.seed(self.al_config.get('random_seed', 42))
        num_classes = len(self.class_names)
        
        # Only consider samples that are marked as available in train_mask
        available_indices = np.where(train_mask == 1)[0]
        if len(available_indices) == 0:
            return classifier, np.zeros_like(train_mask)
        
        # Get labels for available samples only
        available_labels = np.array([train_dset.samples[i].label for i in available_indices])
        
        # Group available samples by class
        class_indices = {}
        for i in range(num_classes):
            class_mask = available_labels == i
            class_indices[i] = available_indices[class_mask]
        
        mask_copy = np.zeros_like(train_mask)
        random_fraction = self.al_config.get('percentage', 0.1)
        
        for i, indices in class_indices.items():
            if len(indices) > 0:  # Only process classes that have samples
                # Calculate number of samples to select from this class
                n_select = max(1, int(len(indices) * random_fraction))  # Select at least 1 if class has samples
                n_select = min(n_select, len(indices))  # Don't select more than available
                
                # Randomly select samples from this class
                random_indices = np.random.choice(indices, size=n_select, replace=False)
                mask_copy[random_indices] = 1
        
        return classifier, mask_copy

class KMeans:
    """KMeans implementation for clustering.
    """
    def __init__(self, n_clusters=5, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit_predict(self, X):
        n_samples, n_features = X.shape
        
        # Handle edge cases
        if self.n_clusters >= n_samples:
            # If we want more clusters than samples, just return each sample as its own cluster
            return np.arange(n_samples)
        
        # Randomly initialize centroids
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()

        for it in range(self.max_iters):
            # Assign clusters - compute distances in smaller chunks to save memory
            labels = np.zeros(n_samples, dtype=int)
            chunk_size = min(20, n_samples)  # Process in smaller chunks
            
            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                X_chunk = X[start_idx:end_idx]
                
                # Compute distances for this chunk
                distances = np.linalg.norm(X_chunk[:, np.newaxis] - self.centroids[np.newaxis, :], axis=2)
                labels[start_idx:end_idx] = np.argmin(distances, axis=1)
                del distances  # Free memory immediately
                del X_chunk

            # Compute new centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]  # Keep old centroid if no points assigned

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids
        
        del new_centroids

        return labels

class ALKMS(ALModule):
    """KMeans-based Active Learning module.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        available_idx = np.where(train_mask == 1)[0]
        if available_idx.size == 0:
            return classifier, np.zeros_like(train_mask)
        dset = copy.deepcopy(train_dset)
        dset.transform = eval_dset.transform
        loader = DataLoader(dset, batch_size=int(self.al_config.get("stream_batch_size", 256)),
                            shuffle=False, num_workers=4, pin_memory=False)

        # Global budget: percentage * dset size
        pct = float(self.al_config.get("percentage", 0.1))
        per_cluster = int(self.al_config.get("num_sample_per_cluster", 1))
        budget_total = max(1, int(round(pct * len(available_idx))))
        picked = []
        cursor = 0

        for inputs, *_ in loader:
            if len(picked) >= budget_total:
                break
            bsz = inputs.size(0)
            batch_gidx = available_idx[cursor:cursor+bsz]
            cursor += bsz

            inputs = inputs.to(self.device, non_blocking=True)
            with torch.no_grad():
                _logits, feats = classifier.forward(inputs, return_feats=True)
            feats_np = feats.cpu().numpy()
            del feats, inputs

            batch_budget = min(budget_total - len(picked), max(1, int(round(pct * bsz))))
            k = max(1, min(len(feats_np), batch_budget // max(1, per_cluster)))

            kmeans = KMeans(n_clusters=int(k))
            labels = kmeans.fit_predict(feats_np)

            # For each cluster, choose points closest to the centroid
            chosen_in_batch = []
            for c in range(k):
                idxs = np.where(labels == c)[0]
                if idxs.size == 0:
                    continue
                cluster_feats = feats_np[idxs]
                centroid = cluster_feats.mean(axis=0, keepdims=True)
                dists = np.linalg.norm(cluster_feats - centroid, axis=1)
                n_take = min(per_cluster, idxs.size)
                local = idxs[np.argsort(dists)[:n_take]]
                chosen_in_batch.extend(batch_gidx[local])

            if len(chosen_in_batch) > batch_budget:
                # recompute distances for chosen only and keep the closest ones
                sel_mask = np.isin(np.arange(len(feats_np)), np.array(chosen_in_batch) - batch_gidx[0])
                sel_feats = feats_np[sel_mask]
                centroid_all = sel_feats.mean(axis=0, keepdims=True)
                d_all = np.linalg.norm(sel_feats - centroid_all, axis=1)
                keep = np.argsort(d_all)[:batch_budget]
                chosen_in_batch = list(np.array(chosen_in_batch)[keep])

            picked.extend([int(x) for x in chosen_in_batch])
        new_mask = np.zeros_like(train_mask)
        new_mask[np.array(picked, dtype=int)] = 1
        return classifier, new_mask


@torch.no_grad()
def gradnorm_uniform_L1(logits, feats, T=1.0):
    p = torch.softmax(logits / T, dim=1)                  # [N, C]
    C = p.size(1)
    out_term = torch.sum(torch.abs(1.0 - C * p), dim=1)   # [N]
    # feature term: L1 norm of last-layer input features
    feat_term = torch.sum(torch.abs(feats), dim=1)        # [N]
    return (feat_term * out_term) / (C * T + 1e-12)       # [N]

class ALALOE(ALModule):
    """
    ALOE: cluster-first, then OOD with GradNorm thresholding (95% TPR), then pick top-B.
    Paper: two-stage selection with k-means diversity followed by OOD prioritization.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        available_idx = np.where(train_mask == 1)[0]
        if available_idx.size == 0:
            return classifier, np.zeros_like(train_mask)

        dset = copy.deepcopy(train_dset)
        dset.transform = eval_dset.transform
        loader = DataLoader(dset, batch_size=int(self.al_config.get("stream_batch_size", 256)),
                            shuffle=False, num_workers=4, pin_memory=False)

        pct = float(self.al_config.get("percentage", 0.1))
        per_cluster = int(self.al_config.get("num_sample_per_cluster", 1))
        budget_total = max(1, int(round(pct * len(available_idx))))
        T = float(self.al_config.get("temperature", 1.0))
        # Direction knob: GradNorm paper ⇒ ID higher ⇒ OOD if score < τ (5th pct for 95% TPR)
        id_higher = bool(self.al_config.get("gradnorm_id_is_higher", True))
        ood_pct = float(self.al_config.get("aloe_ood_percentile", 5.0 if id_higher else 95.0))

        picked = []
        cursor = 0

        with torch.inference_mode():
            for inputs, *_ in loader:
                if len(picked) >= budget_total:
                    break
                bsz = inputs.size(0)
                batch_gidx = available_idx[cursor:cursor+bsz]
                cursor += bsz

                inputs = inputs.to(self.device, non_blocking=True)
                with torch.no_grad():
                    logits, feats = classifier.forward(inputs, return_feats=True)
                # import pdb; pdb.set_trace()
                feats_np = feats.cpu().numpy()
                scores = gradnorm_uniform_L1(logits.float(), feats.float(), T=T).cpu().numpy()

                # Per-batch k and budget
                batch_budget = min(budget_total - len(picked), max(1, int(round(pct * bsz))))
                k = max(1, min(len(feats_np), batch_budget // max(1, per_cluster)))

                # per-batch diversity via k-means
                kmeans = KMeans(n_clusters=int(k))
                labels = kmeans.fit_predict(feats_np)

                # per-batch OOD threshold & cluster ranking
                tau = np.percentile(scores, ood_pct)
                is_ood = (scores < tau) if id_higher else (scores > tau)

                # OOD ratio per cluster, tie-break on “more OOD-like” mean score
                cluster_stats = []
                for c in range(k):
                    idxs = np.where(labels == c)[0]
                    if idxs.size == 0:
                        continue
                    ood_ratio = float(is_ood[idxs].mean())
                    mean_score = float(scores[idxs].mean())
                    tie = -mean_score if id_higher else mean_score
                    cluster_stats.append((c, ood_ratio, tie))

                # Choose clusters with the largest OOD ratio
                cluster_stats.sort(key=lambda t: (t[1], t[2]), reverse=True)
                clusters_to_take = min(len(cluster_stats), math.ceil(batch_budget / max(1, per_cluster)))
                chosen_clusters = [c for c, _, _ in cluster_stats[:clusters_to_take]]

                # From each chosen cluster, take the most OOD-like samples
                chosen_in_batch = []
                for c in chosen_clusters:
                    idxs = np.where(labels == c)[0]
                    if idxs.size == 0: 
                        continue
                    order = np.argsort(scores[idxs])
                    order = order[:per_cluster] if id_higher else order[::-1][:per_cluster]
                    local = idxs[order]
                    chosen_in_batch.extend(batch_gidx[local])

                if len(chosen_in_batch) > batch_budget:
                    # Keep the most OOD-like overall
                    local_scores = scores[np.array(chosen_in_batch) - batch_gidx[0]]
                    keep = np.argsort(local_scores)[:batch_budget] if id_higher else np.argsort(-local_scores)[:batch_budget]
                    chosen_in_batch = list(np.array(chosen_in_batch)[keep])

                del labels, kmeans, feats_np, feats, logits, inputs  # Free memory immediately

                picked.extend([int(x) for x in chosen_in_batch])

        new_mask = np.zeros_like(train_mask)
        new_mask[np.array(picked, dtype=int)] = 1
        return classifier, new_mask

def align_loss(feats, target):
    feats = F.normalize(feats, dim=-1)
    return -(feats @ target.unsqueeze(1)).squeeze(1).mean()

class ALFeatureResonance(ALModule):
    """Feature Resonance (FR) for next-ckp OOD detection.
    Returns: classifier (unchanged), FR_mask over next ckp's train set (1 = OOD-like).
    """
    def __init__(self, al_config, common_config, class_names, args, device):
        self.id_eval_dset = []     # optional: not used unless you wire update_eval_dset
        self.ood_eval_dset = []
        self.al_config = al_config
        self.common_config = common_config
        self.class_names = class_names
        self.args = args
        self.device = device

    def _make_random_target(self, feat_dim, seed=42):
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.al_config.get('fr_seed', seed)))
        v = torch.randn(feat_dim, generator=g, device=self.device)
        v = v / (v.norm(p=2) + 1e-12)
        return v

    @torch.no_grad()
    def _extract_feats(self, classifier, dset, batch_size=128):
        if dset is None or len(dset) == 0:
            return torch.empty(0).cpu()
        _logits, feats = extract_logits_features(classifier, dset, self.device, batch_size=batch_size)
        return feats.cpu()

    # ---------- main ----------
    def process(self, classifier, id_train_dset, ood_eval_dset, train_mask, ckp,
                                  id_eval_dset=None, next_train_dset=None):
        """
            (classifier, FR_mask) where FR_mask is a boolean np.array of len(next_train_dset)
            with True indicating OOD-like (small τ).
        """
        assert next_train_dset is not None, "Provide next_train_dset=<next ckp train dataset>"
        if id_train_dset is None:
            logging.warning("(FR)Train all data in first ckp.")
            return classifier, np.ones(len(next_train_dset), dtype=bool)

        # Hyperparams (light-weight micro-step)
        fr_epochs   = int(self.al_config.get('fr_epochs', 10))
        fr_bs    = int(self.al_config.get('fr_batch_size', 64))
        fr_patience = int(self.al_config.get('fr_patience', 2)) # break after N non-improving epochs

        feat_dim = classifier.visual_model.output_dim
        target_v = self._make_random_target(feat_dim)

        # 1) Clone classifier and take a tiny "feature alignment" step on ID
        clf = copy.deepcopy(classifier).to(self.device)
        clf.train()
        # Only update visual backbone; freeze other parts
        for p in clf.parameters():
            p.requires_grad = False
        for p in clf.visual_model.parameters(): 
            p.requires_grad = True

        opt = get_optimizer(clf, self.common_config['optimizer_name'], self.common_config['optimizer_params'])
        id_loader = DataLoader(id_train_dset, batch_size=fr_bs, shuffle=False, num_workers=4, pin_memory=False)

        best_auroc = -1.0
        best_epoch = -1
        no_improve = 0
        tau_id_star   = None
        tau_next_star = None
        prev_snapshot = copy.deepcopy(clf).to(self.device)  # θ_t

        for t in range(fr_epochs):
            # one epoch of resonance training on ID train
            clf.train()
            for images, *_ in id_loader:
                images = images.to(self.device, non_blocking=True)
                _, f = clf.forward(images, return_feats=True)
                loss = align_loss(f, target_v.to(f.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            clf.eval()

            # micro movement τ = ||h_{t+1} - h_t|| on eval splits (ID vs OOD)
            f_id_pre,   f_id_post   = self._extract_feats(prev_snapshot, id_eval_dset),  self._extract_feats(clf, id_eval_dset)
            f_ood_pre,  f_ood_post  = self._extract_feats(prev_snapshot, ood_eval_dset), self._extract_feats(clf, ood_eval_dset)
            tau_id  = (f_id_post  - f_id_pre).norm(p=2, dim=1).numpy()
            tau_ood = (f_ood_post - f_ood_pre).norm(p=2, dim=1).numpy()

            # AUROC: positives = ID (τ larger), negatives = OOD
            if tau_id.size and tau_ood.size:
                y = np.concatenate([np.ones_like(tau_id), np.zeros_like(tau_ood)])
                s = np.concatenate([tau_id, tau_ood])
                auroc_t = float(roc_auc_score(y, s))
            else:
                auroc_t = 0.5

            if auroc_t > best_auroc:
                best_auroc = auroc_t
                best_epoch = t
                no_improve = 0

                # also compute τ on next-train for this new best epoch
                f_next_pre, f_next_post = self._extract_feats(prev_snapshot, next_train_dset), self._extract_feats(clf, next_train_dset)
                tau_next_star = (f_next_post - f_next_pre).norm(p=2, dim=1).numpy()
                tau_id_star   = tau_id  # NOTE: ID-eval τ at best epoch (used for γ)
            else:
                no_improve += 1
                if no_improve >= fr_patience:
                    break

            logging.info(f"FR: Epoch {t+1}/{fr_epochs}, AUROC={auroc_t:.4f}, Best AUROC={best_auroc:.4f} at epoch {best_epoch+1}")
            # advance snapshot: θ_t ← θ_{t+1}
            del prev_snapshot  # Explicitly delete the old model to free memory
            prev_snapshot = copy.deepcopy(clf).to(self.device)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logging.info(f"FR: best AUROC={best_auroc:.4f} at epoch {best_epoch+1}")

        # ---- set γ from ID-eval at the best epoch (95% ID TPR) ----
        assert tau_id_star is not None and tau_next_star is not None, "FR failed to record best epoch stats."
        gamma = float(np.percentile(tau_id_star, (1.0 - 0.95) * 100.0))  # 5th percentile if id_tpr=0.95

        # ---- mask next-train: True = OOD-like (small τ) ----
        FR_mask = (tau_next_star <= gamma)

        del clf, prev_snapshot, f_id_pre, f_id_post, f_ood_pre, f_ood_post
        return classifier, FR_mask         

class SampleModel(nn.Module):
    """Sample model for ActiveFT.
    """
    def __init__(self, features, sample_num, temperature, init, distance,
                 balance=1.0):
        super(SampleModel, self).__init__()
        self.features = copy.deepcopy(features)
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance

        self.init = init
        self.distance = distance

        centroids = self.init_centroids()
        self.centroids = nn.Parameter(centroids).cuda()
        logging.info(f'Initialized centroids shape: {self.centroids.shape}. ')

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(
                    sample_ids, self.sample_num)
        centroids = self.features[sample_ids].clone()
        # centroids = nn.Parameter(centroids).cuda()
        return centroids

    def get_loss(self):
        centroids = F.normalize(self.centroids, dim=1)
        prod = torch.matmul(self.features, centroids.transpose(1, 0))  # (n, k)
        prod = prod / self.temperature
        prod_exp = torch.exp(prod)
        prod_exp_pos, pos_k = torch.max(prod_exp, dim=1)  # (n, )

        cent_prod = torch.matmul(centroids.detach(), centroids.transpose(1, 0))  # (k, k)
        cent_prod = cent_prod / self.temperature
        cent_prod_exp = torch.exp(cent_prod)
        cent_prob_exp_sum = torch.sum(cent_prod_exp, dim=0)  # (k, )

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)
        return J
    
    def get_sample_ids(self):
        centroids = self.centroids.clone().detach()
        centroids = F.normalize(centroids, dim=1)
        dist = centroids @ self.features.transpose(1, 0)
        sample_ids = set()
        _, ids_sort = torch.sort(dist, dim=1, descending=True)
        for i in range(ids_sort.shape[0]):
            for j in range(ids_sort.shape[1]):
                if ids_sort[i, j].item() not in sample_ids:
                    sample_ids.add(ids_sort[i, j].item())
                    break
        sample_ids = np.array(list(sample_ids), dtype=np.int32)
        sample_ids = np.sort(sample_ids)
        logging.info(f'Number of selected samples: {len(sample_ids)}. ')
        return sample_ids


class ALActiveFT(ALModule):
    """ActiveFT model for selecting samples based on features.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        # Extract features
        train_dset_noaug = copy.deepcopy(train_dset)
        train_dset_noaug.transform = eval_dset.transform
        _, features = extract_logits_features(classifier, train_dset_noaug, self.device)
        if len(features) == 0:
            return classifier, copy.deepcopy(train_mask)
        
        # Filter features based on train_mask
        features_subset = features[train_mask == 1]
        features_subset_indices = np.where(train_mask == 1)[0]
        
        # Get parameters
        percentage = self.al_config.get('percentage', 0.1)
        sample_num = int(percentage * len(train_dset))
        temperature = self.al_config.get('temperature', 0.07)
        init = self.al_config.get('init', 'random')
        distance = self.al_config.get('distance', 'euclidean')
        balance = self.al_config.get('balance', 1.0)
        
        # Init model
        sample_model = SampleModel(
            features=features_subset,
            sample_num=sample_num,
            temperature=temperature,
            init=init,
            distance=distance,
            balance=balance
        ).to(self.device)
        
        # Train the model
        max_iter = 300
        optimizer = optim.Adam(sample_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=1e-6)

        for i in range(max_iter):
            loss = sample_model.get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            if i % 50 == 0:
                logging.info(f'ActiveFT iter {i}, loss: {loss.item():.10f}, lr: {lr:.8f}. ')
        sample_ids = sample_model.get_sample_ids()
        # ######### DEBUG, uncomment if needed #########
        # debug_save_file = os.path.join(self.args.save_dir, 'activeft_out', f'{ckp}.pkl')
        # features_subset = sample_model.features.cpu().numpy()
        # os.makedirs(os.path.dirname(debug_save_file), exist_ok=True)
        # with open(debug_save_file, 'wb') as f:
        #     pickle.dump({
        #         'features_subset': features_subset,
        #         'sample_ids': sample_ids
        #     }, f)
        # #########################
        activeft_mask = np.zeros(train_mask.shape, dtype=train_mask.dtype)
        for i, ind in enumerate(features_subset_indices):
            if i in sample_ids:
                activeft_mask[ind] = 1
        return classifier, activeft_mask


class ALMLS(ALModule):
    """Select samples with minimum maximum logit score.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        # Extract features
        train_dset_noaug = copy.deepcopy(train_dset)
        train_dset_noaug.transform = eval_dset.transform
        logits, _ = extract_logits_features(classifier, train_dset_noaug, self.device)
        if len(logits) == 0:
            return classifier, copy.deepcopy(train_mask)
        sample_indices = np.where(train_mask == 1)[0]
        # Get parameters
        percentage = self.al_config.get('percentage')
        sample_num = int(percentage * len(train_dset))

        score = torch.amax(logits.float(), dim=1)
        sample_ids = torch.topk(-score, sample_num)[1].cpu().numpy()
        mls_mask = np.zeros(train_mask.shape, dtype=train_mask.dtype)
        for i, ind in enumerate(sample_indices):
            if i in sample_ids:
                mls_mask[ind] = 1
        return classifier, mls_mask

class ALMSP(ALModule):
    """Select samples with minimum maximum softmax probability.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, ckp):
        # Extract features
        train_dset_noaug = copy.deepcopy(train_dset)
        train_dset_noaug.transform = eval_dset.transform
        logits, _ = extract_logits_features(classifier, train_dset_noaug, self.device)
        if len(logits) == 0:
            return classifier, copy.deepcopy(train_mask)
        sample_indices = np.where(train_mask == 1)[0]
        # Get parameters
        percentage = self.al_config.get('percentage')
        sample_num = int(percentage * len(train_dset))

        softmax_logits = F.softmax(logits, dim=1)
        score = torch.amax(softmax_logits.float(), dim=1)
        sample_ids = torch.topk(-score, sample_num)[1].cpu().numpy()
        msp_mask = np.zeros(train_mask.shape, dtype=train_mask.dtype)
        for i, ind in enumerate(sample_indices):
            if i in sample_ids:
                msp_mask[ind] = 1
        return classifier, msp_mask


AL_METHODS = {
    'none': ALNone,
    'all': ALAll,
    'random': ALRandom,
    'cls_random': ALClsRandom,
    'activeft': ALActiveFT,
    'mls': ALMLS,
    'msp': ALMSP,
    'kms': ALKMS,
    'aloe': ALALOE,
    'fr': ALFeatureResonance,  # Deprecated, use 'activeft' instead
}


def get_al_module(al_config, common_config, class_names, args, device):
    """Entry point for getting the active learning module.
    
    Args:
        al_config (dict): From the config file.
        common_config (dict): From the config file.
        class_names (list): List of class names. From the config file.
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Device to use.
    Returns:
        ALModule: An instance of the active learning module.
    
    """
    method = al_config.get('method', 'none')
    if method not in AL_METHODS:
        raise ValueError(f'Unknown AL method: {method}.')
    return AL_METHODS[method](al_config, common_config, class_names, args, device)
