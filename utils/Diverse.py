import os
import random

import torch
import numpy as np


def init_and_seed(config):
    os.environ['PYTHONHASHSEED'] = str(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(config['torch_num_threads'])
    '''
    This makes the experiments fully reproducible, but can impact performance
    so it's not activate by default
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    '''


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device, non_blocking=True)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device, non_blocking=True)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device, non_blocking=True)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device, non_blocking=True)