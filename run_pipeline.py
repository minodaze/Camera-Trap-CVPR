import argparse
import logging
import os
import copy
import time

from datetime import datetime
import numpy as np
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader
import pprint
import pickle

from core import *
from core.module import get_al_module, get_cl_module, get_ood_module


def parse_args():
    """Parse command-line arguments for the adaptive learning pipeline.
        
        Returns:
            args (argparse.Namespace): Parsed command-line arguments.
    
    """
    # Configurations
    parser = argparse.ArgumentParser(description='Adaptive Workflow')
    parser.add_argument('--c', type=str, help='Configuration file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--seed', type=int, default=9527, help='Random seed')
    parser.add_argument('--eval_per_epoch', action='store_true', help='Evaluate per epoch')
    parser.add_argument('--no_save', action='store_true', help='Do not save model')
    parser.add_argument('--eval_only', action='store_true', help='Evaluate only')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    # Override configurations
    if args.c:
        with open(args.c, 'r') as f:
            yml = yaml.YAML(typ='rt')
            config = yml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    args.gpu_id = None
    return args

def setup_logging(log_path, debug):
    """Setup logging for the training process.
    
        Args:
            log_path (str): Path to save logs.
            debug (bool): Whether to run in debug mode.
        Returns:
            log_path (str): Path to save logs.
    
    """
    # Setup logging
    logger = logging.getLogger()
    if not debug:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_path, 'log')
    else:
        logger.setLevel(logging.DEBUG)
        curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        log_path = os.path.join(log_path, f'debug_{curr_time}')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Log to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Log to file
    if log_path:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        else:
            raise ValueError(f'Log path {log_path} already exists. ')
        log_file = os.path.join(log_path, 'log.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return log_path

def pretrain(classifier, class_names, pretrain_config, common_config, device):
    """Pretrain the classifier on the pretraining dataset.
        
        Args:
            classifier (nn.Module): Classifier model.
            class_names (list): List of class names.
            pretrain_config (dict): Pretraining configuration.
            common_config (dict): Common configuration.
            device (str): Device to use for training.
        Returns:
            classifier (nn.Module): Classifier after pretraining.
    
    """
    # Get pretrain configurations
    pretrain_data_config_path = pretrain_config['pretrain_data_config_path']
    epochs = pretrain_config['epochs']
    optimizer_name = common_config['optimizer_name']
    optimizer_params = common_config['optimizer_params']
    train_batch_size = common_config['train_batch_size']
    
    # Get optimizer
    optimizer = get_optimizer(classifier, optimizer_name, optimizer_params)
    
    # Get Scheduler
    if common_config['scheduler'] is not None:
        scheduler = get_scheduler(optimizer, common_config['scheduler'], common_config['scheduler_params'])
    else:
        scheduler = None
    
    # Get dataset
    dataset = CkpDataset(pretrain_data_config_path, class_names)
    dataset = dataset.get_subset(is_train=True, ckp_list="ckp_-1")
    logging.info(f'Pretrain dataset size: {len(dataset)}. ')
    
    # Get loss function
    f_loss = get_f_loss(
        pretrain_config['loss_type'], 
        dataset.samples, 
        len(class_names),
        device,
        alpha=pretrain_config.get('loss_alpha', None),
        beta=pretrain_config.get('loss_beta', None),
        gamma=pretrain_config.get('loss_gamma', None)
    )
    
    # Get dataloader
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    
    # Train
    train(classifier, optimizer, loader, epochs, device, f_loss, scheduler=scheduler)
    return classifier

def run(args):
    """Main execution workflow for the adaptive learning pipeline.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        Returns:
            None
    
    """
    # Print args
    logging.info(pprint.pformat(vars(args)))
    common_config = args.common_config
    pretrain_config = args.pretrain_config
    ood_config = args.ood_config
    al_config = args.al_config
    cl_config = args.cl_config
    class_names = args.class_names
    
    is_crop = True if cl_config['method'] == 'co2l' else False
    
    # Load model
    classifier = build_classifier(common_config['model'], class_names, args.device)
    
    # Pretrain
    if pretrain_config['pretrain']:
        logging.info('Pretraining classifier... ')
        classifier = pretrain(classifier, 
                              class_names, 
                              pretrain_config, 
                              common_config, 
                              args.device)
    else:
        logging.info('Skipping pretraining... ')
    
    # Prepare dataset
    train_dset = CkpDataset(common_config["train_data_config_path"], class_names, is_crop=is_crop)
    eval_dset = CkpDataset(common_config["eval_data_config_path"], class_names)
    
    # Print ckp dict
    ckp_list = train_dset.get_ckp_list()
    logging.info(f'Checkpoint list, length: {len(ckp_list)}: ')
    for ckp in ckp_list:
        logging.info(f'\t{ckp} ')
    
    # Initialize modules
    ood_module = get_ood_module(ood_config, common_config, class_names, args, args.device)
    al_module = get_al_module(al_config, common_config, class_names, args, args.device)
    cl_module = get_cl_module(classifier, cl_config, common_config, class_names, args, args.device)
    
    # Main loop
    for i in range(len(ckp_list)):
        # Get checkpoint
        ckp_prev = ckp_list[i - 1] if i > 0 else None
        ckp = ckp_list[i]
        logging.info(f'Training on checkpoint {ckp}. ')
        
        # Get training and evaluation dataset
        ckp_train_dset = train_dset.get_subset(is_train=True, ckp_list=ckp_prev)
        ckp_eval_dset = eval_dset.get_subset(is_train=False, ckp_list=ckp)
        logging.info(f'Training dataset size: {len(ckp_train_dset)}. ')
        logging.info(f'Evaluation dataset size: {len(ckp_eval_dset)}. ')
        
        # Initialize mask
        train_dset_mask = np.ones(len(ckp_train_dset), dtype=bool)
        
        # Run OOD detection
        classifier, ood_mask = ood_module.process(classifier, ckp_train_dset, ckp_eval_dset, train_dset_mask)
        logging.info(f'OOD mask: {ood_mask.sum()} / {len(ood_mask)}. ')
        
        # Run active learning
        classifier, al_mask = al_module.process(
            classifier, 
            ckp_train_dset, 
            ckp_eval_dset, 
            ood_mask, 
            ckp=ckp
        )
        logging.info(f'AL mask: {al_mask.sum()} / {len(al_mask)}. ')

        # Prepare evaluation dataloader
        cl_eval_loader = DataLoader(ckp_eval_dset, batch_size=common_config['eval_batch_size'], shuffle=False)

        # Run continual learning
        classifier = cl_module.process(
            classifier, 
            ckp_train_dset, 
            ckp_eval_dset, 
            al_mask, 
            eval_per_epoch=args.eval_per_epoch, 
            eval_loader=cl_eval_loader, 
            ckp=ckp
        )

        # Save model and predictions
        loss_arr, preds_arr, labels_arr = eval(classifier, cl_eval_loader, args.device, chop_head=common_config['chop_head'])
        print_metrics(loss_arr, preds_arr, labels_arr, len(class_names))
        if not args.no_save:
            logging.info(f'Saving model to {args.save_dir}. ')
            save_path = os.path.join(args.save_dir, f'{ckp}.pth')
            torch.save(classifier.state_dict(), save_path)
        logging.info(f"Saving predictions to {args.save_dir}. ")
        pred_path = os.path.join(args.save_dir, f'{ckp}_preds.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump((preds_arr, labels_arr), f)
        mask_path = os.path.join(args.save_dir, f'{ckp}_mask.pkl')
        with open(mask_path, 'wb') as f:
            pickle.dump((ood_mask, al_mask), f)

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = args.log_path
    if args.debug:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = os.path.join(args.log_path, f"debug-{ts}")

    # Setup logging
    args.save_dir = setup_logging(args.log_path, args.debug)
    logging.info(f'Saving to {save_dir}. ')

    # Save configuration
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yml = yaml.YAML()
        yml.dump(vars(args), f)

    # Run
    start_time = time.time()
    run(args)
    end_time = time.time()

    # Print elapsed time
    logging.info(f'Elapsed time: {end_time - start_time:.2f} seconds. ')
