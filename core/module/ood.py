import copy
import numpy as np
import torch
from abc import ABC, abstractmethod
from ..common import eval

"""

ood.py

Out-of-Distribution (OOD) detection module for the pipeline.
---------------------------------------------------------

This module contains classes and methods for handling OOD detection. Including an abstract base class `OODModule` and concrete implementations for different OOD methods:

    -- `OODAll`: No OOD detection, all samples are considered out-of-distribution and selected.
    -- `OODNone`: No OOD detection, all samples are considered in-distribution and not selected.
    -- `OODOracle`: OOD detection using oracle labels, where the model is evaluated on the training dataset and the incorrect samples are selected.

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

    @abstractmethod
    def process(self, classifier, train_dset, eval_dset, train_mask):
        pass


class OODAll(OODModule):
    """No OOD detection, all samples are considered out-of-distribution and selected.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask):
        return classifier, copy.deepcopy(train_mask)


class OODNone(OODModule):
    """No OOD detection, all samples are considered in-distribution and not selected.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask):
        return classifier, np.zeros_like(train_mask)


class OODOracle(OODModule):
    """OOD detection using oracle labels, where the model is evaluated on the training dataset and the incorrect samples are selected.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask):
        train_dset.eval()
        loader = torch.utils.data.DataLoader(train_dset, batch_size=self.common_config['eval_batch_size'], shuffle=False)
        _, preds_arr, labels_arr = eval(classifier, loader, self.device)
        train_dset.train()
        correct_mask = (preds_arr == labels_arr)
        mask_copy = copy.deepcopy(train_mask)
        mask_copy[correct_mask] = 0
        return classifier, mask_copy


OOD_METHODS = {
    'all': OODAll,
    'none': OODNone,
    'oracle': OODOracle
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
