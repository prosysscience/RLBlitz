from torch import optim, nn

from configs import general_config
from exploration.SoftmaxCategorical import SoftmaxCategorical
from scheduler.Constant import Constant

default_a2c_config = {
    'num_steps': 32,
    'gamma': 0.99,
    'hidden_size': [128, 128],
    'activation_fn': nn.ReLU(),
    'optimizer': optim.Adam,
    'lr': Constant('1e-4'),
    'exploration': SoftmaxCategorical,
}

# we import the default configs
default_a2c_config.update(general_config.default_config)
