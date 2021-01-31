from torch import optim

import general

import torch.nn.functional as F
from exploration.SoftmaxCategorical import SoftmaxCategorical
from scheduler.Constant import Constant

default_a2c_config = {
    'num_steps': 32,
    'gamma': 0.99,
    'hidden_size': [128, 128],
    'activation_fn': F.relu,
    'optimizer': optim.Adam,
    'lr': Constant('1e-4'),
    'exploration': SoftmaxCategorical,
}

# we import the default configs
default_a2c_config.update(general.default_config)
