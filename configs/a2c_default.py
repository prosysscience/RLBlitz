import torch
from torch import optim, nn

from configs import general_config
from exploration.SoftmaxCategorical import SoftmaxCategorical

default_a2c_config = {
    'num_steps': 32,
    'gamma': 0.99,
    'hidden_size': [128, 128],
    'activation_fn': nn.ReLU(),
    'optimizer': optim.Adam,
    'lr_initial': 1e-4,
    # default scheduler is constant, x represent the optimizer
    'lr_scheduler': lambda x: torch.optim.lr_scheduler.StepLR(x, step_size=100, gamma=1.0),
    'exploration': SoftmaxCategorical,
}

# we import the default configs
default_a2c_config.update(general_config.default_config)
