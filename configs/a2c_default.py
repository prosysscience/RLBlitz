from torch import nn
from torch.distributions import Categorical

from configs import general_config
from models.ActorCritic import ActorCritic

default_a2c_config = {
    'distribution': Categorical,
    # Lamdba-GAE: https://arxiv.org/abs/1506.02438
    'use_gae': True,
    'lambda_gae': 0.95,
    # only valuable with large batches
    'normalize_advantage': False,
    'vf_coeff': 0.5,
    'entropy_coeff': 1e-5,
    # neural network
    # keep the
    'common_layers': lambda input_size: None,
    # keep the lambda, it's important !
    'critic_layers': lambda input_size: nn.Sequential(
        nn.Linear(input_size, 124),
        nn.ReLU(),
        nn.Linear(124, 124),
        nn.ReLU(),
        nn.Linear(124, 1),
    ),
    # keep the lambda, it's important !
    'actor_layers': lambda input_size, output_size: nn.Sequential(
        nn.Linear(input_size, 124),
        nn.ReLU(),
        nn.Linear(124, 124),
        nn.ReLU(),
        nn.Linear(124, output_size),
        nn.Softmax(dim=1)
    ),
    # needs to inherit from AbstractActorCritic, in general don't touch this except if you need very specific behavior
    'nn_template': ActorCritic,
}

# we import the default configs
default_a2c_config.update(general_config.default_config)
