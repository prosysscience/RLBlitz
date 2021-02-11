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
    '''
    if config['cudnn_trade_perf_for_reproducibility']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class MockLRScheduler:

    def __init__(self, lr):
        self.lr = lr

    def step(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def get_last_lr(self):
        return [self.lr]
