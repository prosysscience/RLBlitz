import os
import random

import torch
import numpy as np
from torch import nn

from utils import ParameterScheduler
from utils.ParameterScheduler import ConstantScheduler


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


def default_actor_critic(x, gain=np.sqrt(2)):
    return nn.init.orthogonal_(x, gain=gain)


def init_weights(m, function_hidden=default_actor_critic, bias_hidden=0.0,
                 function_output=default_actor_critic, bias_output=0.1):
    linear_layers = [module for module in m.modules() if isinstance(module, nn.Linear)]
    for layers in linear_layers[:-1]:
        function_hidden(layers.weight)
        nn.init.constant_(layers.bias, bias_hidden)
    function_output(linear_layers[-1].weight)
    nn.init.constant_(linear_layers[-1].bias, bias_output)


def set_attr(obj, attr_dict):
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj


def parse_scheduler(spec):
    if not isinstance(spec, dict):
        return ConstantScheduler(spec)
    scheduler = getattr(ParameterScheduler, spec['scheduler_name'])
    set_attr(scheduler, dict(
        start_val=np.nan,
    ))
    set_attr(scheduler, spec)