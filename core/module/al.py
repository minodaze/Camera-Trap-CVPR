import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import copy
from torch.utils.data import DataLoader
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


def extract_logits_features(classifier, dset, device):
    """Utility function to extract logits and features from the classifier.
    """
    classifier.eval()
    with torch.no_grad():
        loader = DataLoader(dset, batch_size=128, shuffle=False)
        features, logits = [], []
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _logits, feature = classifier.forward(inputs, return_feats=True)
            logits.append(_logits.cpu())
            features.append(feature.cpu())
    if len(logits) > 0:
        logits = torch.cat(logits, dim=0).to(device)
        features = torch.cat(features, dim=0).to(device)
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
    def __init__(self, al_config, ood_config, class_names, args, device):
        self.al_config = al_config
        self.ood_config = ood_config
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
        percentage = self.al_config.get('percentage')
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
    'activeft': ALActiveFT,
    'mls': ALMLS,
    'msp': ALMSP,
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
