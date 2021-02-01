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
