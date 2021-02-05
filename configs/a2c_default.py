from torch import nn
from torch.distributions import Categorical

from configs import general_config

default_a2c_config = {
    'logistic_function': nn.Softmax(dim=1),
    'distribution': Categorical,
    # Lamdba-GAE: https://arxiv.org/abs/1506.02438
    'use_gae': True,
    'lambda_gae': 0.95,
    # only valuable with large batches
    'normalize_advantage': False,
    'vf_coeff': 0.5,
    'entropy_coeff': 1e-5,
}

# we import the default configs
default_a2c_config.update(general_config.default_config)
