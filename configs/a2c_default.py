import general

import torch.nn.functional as F
from exploration.SoftmaxCategorical import SoftmaxCategorical

default_a2c_config = {
    'gamma': 0.99,
    'hidden_size': [128, 128],
    'activation_fn': F.relu,
    'exploration': SoftmaxCategorical,
}

# we import the default configs
default_a2c_config.update(general.default_config)
