import torch
from torch import optim, nn
from torch.distributions import Categorical

from configs import general_config

default_a2c_config = {
    'hidden_size': [64, 64],
    'activation_fn': nn.ReLU(),
    'logistic_function': nn.Softmax(dim=1),
    'distribution': Categorical,
    'optimizer': optim.Adam,  # if you need more control, you can define a lambda
    'lr_initial': 1e-4,
    # default scheduler is constant, x represent the optimizer
    'lr_scheduler': lambda x: torch.optim.lr_scheduler.LambdaLR(x, lr_lambda=lambda epoch: 1.0),
    'clip_grad_norm': None,
    # only valuable with large batches
    'normalize_advantage': False,
    'vf_coeff': 0.5,
    'entropy_coeff': 1e-4,
}

# we import the default configs
default_a2c_config.update(general_config.default_config)
