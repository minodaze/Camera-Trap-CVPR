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
from torch.nn import functional as F

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

    def _train(self, classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader, gpu_monitor=None, ckp=None, save_best_model=True, test_per_epoch=False, next_test_loader=None):
        if len(cl_train_dset) == 0:
            logging.info('No samples to train classifier, skipping. ')
        else:
            logging.info(f'Training classifier with {len(cl_train_dset)} samples. ')
            cl_train_loader = DataLoader(cl_train_dset, batch_size=self.common_config['train_batch_size'], shuffle=True, num_workers=12)
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
            
            # Determine model name prefix and save directory
            model_name_prefix = f"ckp_{ckp}" if ckp is not None else "cl_model"
            save_best_model_enabled = getattr(self.args, 'save_best_model', save_best_model)
            save_dir = getattr(self.args, 'save_dir', None) if save_best_model_enabled else None
            
            # Extract test parameters from args if not provided directly
            if not test_per_epoch:
                test_per_epoch = getattr(self.args, 'test_per_epoch', False)
            if next_test_loader is None:
                next_test_loader = getattr(self.args, '_next_test_loader', None)
            
            train(classifier, 
                    optimizer, 
                    cl_train_loader, 
                    self.cl_config['epochs'], 
                    self.device, 
                    f_loss, 
                    eval_per_epoch=eval_per_epoch, 
                    eval_loader=eval_loader,
                    scheduler=scheduler,
                    gpu_monitor=gpu_monitor,
                    save_best_model=save_best_model_enabled,
                    save_dir=save_dir,
                    model_name_prefix=model_name_prefix,
                    validation_mode=getattr(self.args, 'validation_mode', 'balanced_acc'),
                    early_stop_epoch=getattr(self.args, 'early_stop_epoch', 10),
                    test_per_epoch=test_per_epoch,
                    next_test_loader=next_test_loader)

    @abstractmethod
    def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None, gpu_monitor=None):
        pass

    @abstractmethod
    def refresh_buffer(self, new_samples):
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
    def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None, gpu_monitor=None):
        return classifier
    
    def refresh_buffer(self, new_samples):
        pass

    def incremental_step(self, model):
        pass
    
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        pass


class CLNaiveFT(CLModule):
    """Naive fine-tuning. Naively fine-tune the classifier on the new samples."""
    def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None, gpu_monitor=None):
        # Process data
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        # Train
        self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader, gpu_monitor, ckp=ckp, save_best_model=True)
        return classifier

    def refresh_buffer(self, new_samples):
        pass

    def incremental_step(self, model):
        pass

    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        pass

class CLAccumulative(CLModule):
    """Accumulative training. Fine-tune the classifier on all samples seen so far.
    """
    def process(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None, gpu_monitor=None):
        # Process data
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        cl_train_dset.add_samples(self.buffer)
        # Train
        self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader, gpu_monitor, ckp=ckp, save_best_model=True)
        # Process buffer
        self.buffer.extend(train_dset.samples)
        return classifier

    def refresh_buffer(self, new_samples):
        pass

    def incremental_step(self, model):
        pass
    
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        pass

class CLAccumulativeScratch(CLModule):
    """Accumulative training with scratch. Fine-tune the classifier on all samples seen so far, but use a new classifier each time.
    """
    def process(self, _, train_dset, eval_dset, train_mask, eval_per_epoch=True, eval_loader=None, ckp=None, gpu_monitor=None):
        # global idx
        classifier = copy.deepcopy(self._classifier)
        # Process data
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        cl_train_dset.add_samples(self.buffer)
        # Train
        self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader, gpu_monitor, ckp=ckp, save_best_model=True)
        # Process buffer
        for msk, sample in zip(train_mask, train_dset.samples):
            if msk:
                self.buffer.append(sample)
        return classifier
    
    def refresh_buffer(self, new_samples):
        pass

    def incremental_step(self, model):
        pass
    
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        pass

class CLAccumulativeScratchLWF(CLModule):
    """Accumulative training with scratch. Fine-tune the classifier on all samples seen so far, but use a new classifier each time.
    """
    def process(self, _, train_dset, eval_dset, train_mask, eval_per_epoch=True, eval_loader=None, ckp=None, gpu_monitor=None):
        # global idx
        classifier = copy.deepcopy(self._classifier)
        # Set the reference model for distillation BEFORE training
        self.incremental_step(classifier)
        # Process data
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        cl_train_dset.add_samples(self.buffer)
        # Train
        self._train(classifier, cl_train_dset, eval_dset, eval_per_epoch, eval_loader, gpu_monitor, ckp=ckp, save_best_model=True)
        # Process buffer
        for msk, sample in zip(train_mask, train_dset.samples):
            if msk:
                self.buffer.append(sample)
        return classifier
    
    def incremental_step(self, model):
        self.ref_model = self._classifier  # update the reference model
        self.ref_model.eval()              # no training on it, just inference

    def refresh_buffer(self, new_samples):
        pass
    
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        pass

class CLReplay(CLModule):
    """
    CLEAR-style replay:
        • keep a fixed-size, class-balanced buffer
        • on every round mix NEW : REPLAY samples in a 50 : 50 ratio
        • after training update the buffer and re-balance it
    """

    def _sample_from_buffer(self, n, classifier, train_dset):
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
        """Make the  self.buffer class-balanced and ≤ buf_size."""
        if len(self.buffer) == 0:
            return                                            # nothing to rebalance
        by_cls = defaultdict(list)
        for s in self.buffer:
            by_cls[s.label].append(s)

        n_cls     = len(by_cls)
        per_class = max(1, buf_size // n_cls)
        if per_class < 10:
            self.cl_config['buffer_size'] = buf_size*2  # increase the buffer size to keep at least 10 samples per class
            buf_size = self.cl_config.get('buffer_size', 500)
            per_class = max(1, buf_size // n_cls)

        new_buf = []
        for samples in by_cls.values():
            if len(samples) > per_class:                 # down-sample
                new_buf.extend(random.sample(samples, per_class))
            else:                                        # keep them all
                k = per_class - len(samples)
                new_buf.extend(samples)
                new_buf.extend(random.choices(samples, k=k))  # duplicate to fill
        self.buffer = new_buf

        for s in self.buffer:
            s.is_buf = True                     # mark as buffer sample

    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        # update the buffer with **new** data then re-balance it
        for msk, sample in zip(train_mask, train_dset.samples):
            if not sample.is_buf:  # add only new samples, not buffer ones
                self.buffer.append(sample)
        buf_size = self.cl_config.get('buffer_size', 500)
        self._rebalance_buffer(buf_size)                       # trim/balance

    def process(self, classifier, train_dset, eval_dset, train_mask,
                eval_per_epoch=False, eval_loader=None, ckp=None, gpu_monitor=None):

        # 1) collect *new* samples for this round
        cl_train_dset = copy.deepcopy(train_dset)
        cl_train_dset.apply_mask(train_mask)
        new_samples   = cl_train_dset.samples
        n_new         = len(cl_train_dset)

        # refresh the buffer if needed
        self.refresh_buffer(new_samples)

        # 2) replay: draw the same number of samples from the buffer -> expected 50 : 50 ratio in every DataLoader epoch
        replay_samples = self._sample_from_buffer(n_new, classifier, train_dset)
        cl_train_dset.add_samples(replay_samples)

        # 3) incremental step: update the reference model
        self.incremental_step(classifier)

        # 4) train the classifier on NEW ⊕ REPLAY
        self._train(classifier, cl_train_dset,
                    eval_dset, eval_per_epoch, eval_loader, gpu_monitor, ckp=ckp, save_best_model=True)
        
        # 5) after training
        self._after_train(classifier, train_dset, eval_dset, train_mask)

        return classifier

    def incremental_step(self, model):
        pass
    
    def refresh_buffer(self, new_samples):
        pass

class CLLWF(CLReplay):
    """LWF-style replay:
        • keep a fixed-size, class-balanced buffer
        • on every round mix NEW : REPLAY samples in a 50 : 50 ratio
        • after training update the buffer and re-balance it
        • use the reference model to compute the distillation loss
    """
    def incremental_step(self, model):
        self.ref_model = self._classifier  # update the reference model
        self.ref_model.eval()                  # no training on it, just inference

class CLDerpp(CLReplay):
    """DERPP-style replay:
        • keep a fixed-size, class-balanced buffer
        • on every round mix NEW : REPLAY samples in a 50 : 50 ratio
        • after training update the buffer and re-balance it
        • Use mse_loss to provide stability using the old logits.
    """
    def _after_train(self, classifier, train_dset, eval_dset, train_mask, eval_per_epoch=False, eval_loader=None, ckp=None):
        # update the buffer with **new** data then re-balance it
        for msk, sample in zip(train_mask, train_dset.samples):
            self.buffer.append(sample)
        buf_size = self.cl_config.get('buffer_size', 500)
        self._rebalance_buffer(buf_size)                       # trim/balance

        # compute logits for the buffer samples
        if len(self.buffer) == 0:
            logging.info('No samples in buffer, skipping buffer logits computing.')
        else:
            logging.info('Computing logits for the buffer samples')
            cl_buffer_dset = BufferDataset(self.buffer)
            for sample, data in zip(self.buffer, cl_buffer_dset):
                image, label, _, is_buf = data
                image = image.to(self.device)
                if is_buf:
                    with torch.no_grad():
                        logits = classifier(image.unsqueeze(0).to(self.device))[0].cpu()
                        sample.logits = logits

class CLMIR(CLReplay):
    """MIR-style replay:
        • keep a fixed-size, class-balanced buffer
        • select a subset of the buffer samples with the highest scores
        • on every round mix NEW : REPLAY samples in a 50 : 50 ratio
        • after training update the buffer and re-balance it
    """
    def _sample_from_buffer(self, n, classifier, train_dset):
        """Return `n` samples from the buffer picked by the MIR criterion."""
        if len(self.buffer) == 0:
            return []
        
        if n <= len(self.buffer):
            logging.info(f'Selecting {n} samples from the buffer of size {len(self.buffer)} by the MIR criterion.')
            cl_train_loader = DataLoader(train_dset, batch_size=self.common_config['train_batch_size'], shuffle=True, num_workers=12)
            optimizer = get_optimizer(classifier, self.common_config['optimizer_name'], self.common_config['optimizer_params'])
            self._classifier = copy.deepcopy(classifier).to(self.device)
            self._classifier.train()
            # Train one epoch is enough
            for inputs, labels, _, _ in cl_train_loader:
                # Forward
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self._classifier(inputs)
                loss = F.cross_entropy(logits, labels)
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            buf_dset = BufferDataset(self.buffer)
            buf_loader = DataLoader(buf_dset, batch_size=self.common_config['train_batch_size'], shuffle=False, num_workers=12)
            scores = []
            self._classifier.eval()
            classifier.eval()
            with torch.no_grad():
                for imgs, labels, _, _ in buf_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    prev_loss = F.cross_entropy(classifier(imgs), labels, reduction='none').cpu().numpy()
                    curr_loss = F.cross_entropy(self._classifier(imgs), labels, reduction='none').cpu().numpy()
                    scores.append(prev_loss - curr_loss)
            scores = np.concatenate(scores)
            selected_idx = np.argsort(scores)[::-1][:n]  # pick the n highest scores
            selected_samples = [self.buffer[i] for i in selected_idx]
        else:
            # not enough → duplicate some to reach n
            k    = n - len(self.buffer)
            dup  = random.choices(self.buffer, k=k)
            selected_samples = self.buffer + dup
        return selected_samples

class CLRandReplaceOld(CLReplay):
    """
    Replay with *Random Replace Old*:
      • before training, replace <replace_rate>% of the buffer
        entries of every *old* class with fresh images of the
        same class that arrive in the current mini-batch stream.
      • then fall back to ordinary replay (50:50 mix, etc.).
    """
    def refresh_buffer(self, new_samples):
        """new_samples is the list returned by train_dset.samples
           AFTER train_mask is applied.  Each element has .label."""
        r = self.cl_config.get("replace_rate", 0.2)
        if r <= 0 or len(self.buffer) == 0:
            return                                        # nothing to do

        # organise current buffer by class
        old_by_cls = defaultdict(list)
        for idx, s in enumerate(self.buffer):
            old_by_cls[s.label].append(idx)

        # organise incoming new data replacements by class
        new_by_cls = defaultdict(list)
        for s in new_samples:
            if s.label in old_by_cls:                        # only OLD classes
                new_by_cls[s.label].append(s)

        # loop over each old class and swap
        for cls, idx_list in old_by_cls.items():
            n_rep = int(len(idx_list) * r)
            if n_rep == 0 or len(new_by_cls[cls]) == 0:
                continue
            # choose which buffer slots to evict and which new samples to insert
            evict_idx   = random.sample(idx_list, n_rep)
            insert_pool = random.choices(new_by_cls[cls], k=n_rep) if len(new_by_cls[cls]) < n_rep else random.sample(new_by_cls[cls], n_rep)
            # perform in-place replacement
            for buf_pos, new_s in zip(evict_idx, insert_pool):
                self.buffer[buf_pos] = new_s

## CLCO2L: Only useful if these's novel classes, current pipeline does not fit this algorithm.

class CLCO2L(CLReplay):
    """Contrastive Replay (Co2L) with IRD regularisation."""
    def __init__(self, classifier, cl_config, common_config,
                 class_names, args, device):
        super().__init__(classifier, cl_config, common_config,
                         class_names, args, device)

        # ---- SupCon head and loss --------------------------------
        dim_in   = classifier.head.in_features
        proj_dim = self.cl_config.get('proj_dim', 512)
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
            cl_buffer_loader = DataLoader(cl_buffer_dset, batch_size=self.common_config['train_batch_size'], shuffle=True, num_workers=12)
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
                    scheduler=scheduler,
                    train_head_only=True)
        

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
    'accumulative-scratch-lwf': CLAccumulativeScratchLWF,
    'replay': CLReplay,
    'rand-replace-old': CLRandReplaceOld,
    'lwf': CLLWF,
    'mir': CLMIR,
    'derpp' : CLDerpp,
    'co2l': CLCO2L,
    # 'accumulative-scratch-local': CLAccumulativeScratchLocal,
}

def get_cl_module(classifier, cl_config, common_config, class_names, args, device):
    method = cl_config.get('method', 'none')
    if method not in CL_METHODS:
        raise ValueError(f'Unknown AL method: {method}.')
    return CL_METHODS[method](classifier, cl_config, common_config, class_names, args, device)
