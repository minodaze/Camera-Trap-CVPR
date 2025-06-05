import copy
import logging
import numpy as np
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from ..common import get_optimizer, train, eval, get_f_loss, print_metrics, get_scheduler
from ..data import BufferDataset
from abc import ABC, abstractmethod
import random
from collections import defaultdict

"""

cl.py

Continual Learning (CL) module for the pipeline.
---------------------------------------------------------

This mudule contains classes and methods for handling Continual Learning (CL). Including an abstract base class `CLModule` and concrete implementations for different CL methods:
    -- `CLNone`: No training. Used in zero-shot evaluation.
    -- `CLNaiveFT`: Naive fine-tuning. Naively fine-tune the classifier on the new samples.
    -- `CLAccumulative`: Accumulative training. Fine-tune the classifier on all samples seen so far.
    -- `CLAccumulativeScratch`: Accumulative training with scratch. Fine-tune the classifier on all samples seen so far, but use a new classifier each time.
    -- 'CLCLEAR': CLEAR, mix old data and new data in 50:50 ratio for training.

"""


class CLModule(ABC):
    """Abstract base class for CL modules.

        Methods:
            - `_train`: Trains the classifier on the given dataset.

        Abstract methods:
            - `process`: Main function of the CL module. Returns a trained classifier.

    """
    def __init__(self, classifier, cl_config, common_config, class_names, args, device):
        self._classifier = copy.deepcopy(classifier)
        self.cl_config = cl_config
        self.common_config = common_config
        self.class_names = class_names
        self.buffer = []
        self.args = args
        self.device = device
        self.ref_model = None

    def _train(self, classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader):
        if len(cl_train_dset) == 0:
            logging.info('No samples to train classifier, skipping. ')
        else:
            logging.info(f'Training classifier with {len(cl_train_dset)} samples. ')
            cl_train_loader = DataLoader(cl_train_dset, batch_size=self.common_config['train_batch_size'], shuffle=True)
            optimizer = get_optimizer(classifier, self.common_config['optimizer_name'], self.common_config['optimizer_params'])
            if self.common_config['scheduler'] is not None:
                scheduler = get_scheduler(optimizer, self.common_config['scheduler'], self.common_config['scheduler_params'])
            else:
                scheduler = None
            f_loss = get_f_loss(
                self.cl_config['loss_type'], 
                cl_train_loader.dataset.samples, 
                len(self.class_names),
                self.device,
                alpha=self.cl_config.get('loss_alpha', None),
                beta=self.cl_config.get('loss_beta', None),
                gamma=self.cl_config.get('loss_gamma', None),
                ref_model=self.ref_model,
            )
            train(classifier, 
                    optimizer, 
                    cl_train_loader, 
                    self.cl_config['epochs'], 
                    self.device, 
                    f_loss, 
                    eval_per_epoch=eval_per_epoch, 
                    eval_loader=eval_loader,
                    scheduler=scheduler)

    @abstractmethod
    def process(self, classifier, train_dset, eval_dset, train_mask):
        pass

    @abstractmethod
    def incremental_step(self, model):
        pass
    
    @abstractmethod
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        pass

class CLNone(CLModule):
    """No training. Used in zero-shot evaluation.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        return classifier


class CLNaiveFT(CLModule):
    """Naive fine-tuning. Naively fine-tune the classifier on the new samples."""
    def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        # Process data
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        # Train
        self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader)
        return classifier


class CLAccumulative(CLModule):
    """Accumulative training. Fine-tune the classifier on all samples seen so far.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        # Process data
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        cl_train_dset.add_samples(self.buffer)
        # Train
        self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader)
        # Process buffer
        self.buffer.extend(train_dset.samples)
        return classifier


class CLAccumulativeScratch(CLModule):
    """Accumulative training with scratch. Fine-tune the classifier on all samples seen so far, but use a new classifier each time.
    """
    def process(self, _, train_dset, eval_dset, train_mask, eval_per_epoch=True, eval_loader=None, ckp=None):
        # global idx
        classifier = copy.deepcopy(self._classifier)
        # Process data
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        cl_train_dset.add_samples(self.buffer)
        # Train
        self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader)
        # Process buffer
        for msk, sample in zip(train_mask, train_dset.samples):
            if msk:
                self.buffer.append(sample)
        return classifier

class CLReplay(CLModule):
    """
    CLEAR-style replay:
        • keep a fixed-size, class-balanced buffer
        • on every round mix NEW : REPLAY samples in a 50 : 50 ratio
        • after training update the buffer and re-balance it
    """
    def _sample_from_buffer(self, n):
        """Return `n` samples from the buffer (with or without replacement)."""
        if len(self.buffer) == 0:
            return []
        if n <= len(self.buffer):
            return random.sample(self.buffer, n)          # without replacement
        else:
            # not enough → duplicate some to reach n
            k    = n - len(self.buffer)
            dup  = random.choices(self.buffer, k=k)       # with replacement
            return self.buffer + dup

    def _rebalance_buffer(self, buf_size):
        """Make the private  self.buffer  class-balanced and ≤ buf_size."""
        if len(self.buffer) == 0:
            return                                            # nothing to rebalance
        by_cls = defaultdict(list)
        for s in self.buffer:
            by_cls[s.label].append(s)

        n_cls     = len(by_cls)
        per_class = max(1, buf_size // n_cls)

        new_buf = []
        for samples in by_cls.values():
            if len(samples) > per_class:                 # down-sample
                new_buf.extend(random.sample(samples, per_class))
            else:                                        # keep them all
                new_buf.extend(samples)

        # if rounding left us short, pad with random picks (rare)
        while len(new_buf) < min(buf_size, len(self.buffer)):
            new_buf.append(random.choice(self.buffer))

        self.buffer = new_buf[:buf_size]                 # hard cut to limit
    
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        # update the buffer with **new** data then re-balance it
        for msk, sample in zip(train_mask, train_dset.samples):
            self.buffer.append(sample)
        buf_size = self.cl_config.get('buffer_size', 500)
        self._rebalance_buffer(buf_size)                       # trim/balance

    def process(self, classifier, train_dset, eval_dset, train_mask,
                eval_per_epoch=False, eval_loader=None, ckp=None):

        # 1) collect *new* samples for this round
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        n_new         = len(cl_train_dset)

        # 2) replay: draw the same number of samples from the buffer -> expected 50 : 50 ratio in every DataLoader epoch
        replay_samples = self._sample_from_buffer(n_new)
        cl_train_dset.add_samples(replay_samples)

        # 3) incremental step: update the reference model
        self.incremental_step(classifier)

        # 4) train the classifier on NEW ⊕ REPLAY
        self._train(classifier, cl_train_dset,
                    eval_dset, eval_per_epoch, eval_loader)
        
        # 5) after training
        self._after_train(classifier, train_dset, eval_dset, train_mask)  # implement this if needed

        return classifier

class CLLWF(CLReplay):
    """LWF-style replay:
        • keep a fixed-size, class-balanced buffer
        • on every round mix NEW : REPLAY samples in a 50 : 50 ratio
        • after training update the buffer and re-balance it
        • use the reference model to compute the distillation loss
    """
    def incremental_step(self, model):
        self.ref_model = copy.deepcopy(model)  # update the reference model
        self.ref_model.eval()                  # no training on it, just inference

class CLCO2L(CLReplay):
    """Contrastive Replay (Co2L) with IRD regularisation."""
    def __init__(self, classifier, cl_config, common_config,
                 class_names, args, device):
        super().__init__(classifier, cl_config, common_config,
                         class_names, args, device)

        # ---- (a) SupCon head and loss --------------------------------
        dim_in   = self._classifier.head.in_features
        proj_dim = self.cl_config.get('proj_dim', 128)
        proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, proj_dim)
        ).to(device)
        classifier.set_proj_head(proj_head)

        self.ref_model     = None        

    #  incremental-step: copy previous nets so we can distil later
    def incremental_step(self, model):
        self.ref_model = copy.deepcopy(model)  # update the reference model
        self.ref_model.eval()                  # no training on it, just inference

    #  inner training loop
    def _train(self, classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader):
        if len(cl_train_dset) == 0:
            logging.info('No samples to train classifier, skipping. ')
        else:
            logging.info(f'Training classifier with {len(cl_train_dset)} samples. ')
            cl_train_loader = DataLoader(cl_train_dset, batch_size=self.common_config['train_batch_size'], shuffle=True)
            optimizer = get_optimizer(classifier, self.common_config['optimizer_name'], self.common_config['optimizer_params'])
            if self.common_config['scheduler'] is not None:
                scheduler = get_scheduler(optimizer, self.common_config['scheduler'], self.common_config['scheduler_params'])
            else:
                scheduler = None
            f_loss = get_f_loss(
                self.cl_config['loss_type'], 
                cl_train_loader.dataset.samples, 
                len(self.class_names),
                self.device,
                alpha=self.cl_config.get('loss_alpha', None),
                beta=self.cl_config.get('loss_beta', None),
                gamma=self.cl_config.get('loss_gamma', None),
                ref_model=self.ref_model,
            )
            classifier.train()
            epochs = self.cl_config['epochs']
            for epoch in range(epochs):
                classifier.train()
                loss_arr = []
                correct_arr = []
                # for inputs, labels in tqdm(loader):
                for inputs, labels in cl_train_loader:
                    if isinstance(inputs, list):
                        inputs = torch.cat(inputs, dim=0)
                    # Forward
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    bsz = labels.size(0)
                    i1, i2 = torch.split(inputs, [bsz, bsz], dim=0)
                    logits = classifier(i1)
                    preds = logits.argmax(dim=1)
                    correct = preds == labels
                    # SupCon loss
                    proj_features = classifier.proj_features(inputs)
                    loss = f_loss(logits, labels, images=inputs, features=proj_features)
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Append
                    loss_arr.append(loss.cpu().item())
                    correct_arr.append(correct.cpu().numpy())
                loss_arr = np.array(loss_arr)
                correct_arr = np.concatenate(correct_arr, axis=0)
                logging.info(f'Epoch {epoch}, loss: {loss_arr.mean():.4f}, acc: {correct_arr.mean():.4f}, lr: {optimizer.param_groups[0]["lr"]:.8f}. ')

            if scheduler is not None:
                scheduler.step()
            
            if eval_per_epoch:
                loss_arr, preds_arr, labels_arr = eval(classifier, eval_loader, device)
                print_metrics(loss_arr, preds_arr, labels_arr, len(loader.dataset.class_names), log_predix=f'Epoch {epoch}, ')
    
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        # update the buffer with **new** data then re-balance it
        for msk, sample in zip(train_mask, train_dset.samples):
            self.buffer.append(sample)
        buf_size = self.cl_config.get('buffer_size', 500)
        self._rebalance_buffer(buf_size)                       # trim/balance

        # train the classifier with the buffer data
        if len(self.buffer) == 0:
            logging.info('No samples in buffer, skipping buffer training.')
        else:
            logging.info('Train the buffered data')
            cl_buffer_dset = BufferDataset(self.buffer)
            cl_buffer_loader = DataLoader(cl_buffer_dset, batch_size=self.common_config['train_batch_size'], shuffle=True)
            optimizer = get_optimizer(classifier, self.common_config['optimizer_name'], self.common_config['optimizer_params'])
            if self.common_config['scheduler'] is not None:
                scheduler = get_scheduler(optimizer, self.common_config['scheduler'], self.common_config['scheduler_params'])
            else:
                scheduler = None
            f_loss = get_f_loss(
                'ce', 
                cl_buffer_loader.dataset.samples, 
                len(self.class_names),
                self.device,
                alpha=self.cl_config.get('loss_alpha', None),
                beta=self.cl_config.get('loss_beta', None),
                gamma=self.cl_config.get('loss_gamma', None),
                ref_model=self.ref_model,
            )
            train(classifier, 
                    optimizer, 
                    cl_buffer_loader, 
                    self.cl_config['epochs'], 
                    self.device, 
                    f_loss, 
                    eval_per_epoch=eval_per_epoch, 
                    eval_loader=eval_loader,
                    scheduler=scheduler)
        

# No need anymore.
# class CLAccumulativeScratchLocal(CLModule):
#     """Accumulative training with scratch. Fine-tune the classifier on all samples seen so far, but use a new classifier each time.
#     """
#     def _load_accumu_classifier_ckp(self, ckp):
#         ckp_path = self.cl_config["ckp_path"]
#         ckp_path = os.path.join(ckp_path, f"{ckp}.pth")
#         ckp_clssifier = copy.deepcopy(self._classifier)
#         ckp_clssifier.load_state_dict(torch.load(ckp_path))
#         return ckp_clssifier

#     def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
#         classifier = self._load_accumu_classifier_ckp(ckp)
#         # Process data
#         cl_train_dset = copy.deepcopy(train_dset)
#         cl_train_dset.apply_mask(train_mask)
#         # Train
#         self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader)
#         # Process buffer
#         return classifier


CL_METHODS = {
    'none': CLNone,
    'naive-ft': CLNaiveFT,
    'accumulative': CLAccumulative,
    'accumulative-scratch': CLAccumulativeScratch,
    'replay': CLReplay,
    'lwf': CLLWF,
    'co2l': CLCO2L,
    # 'accumulative-scratch-local': CLAccumulativeScratchLocal,
}

def get_cl_module(classifier, cl_config, common_config, class_names, args, device):
    method = cl_config.get('method', 'none')
    if method not in CL_METHODS:
        raise ValueError(f'Unknown AL method: {method}.')
    return CL_METHODS[method](classifier, cl_config, common_config, class_names, args, device)
