import multiprocessing

import psutil
import torch
from torch import nn, optim
from torch.distributions import Categorical

from models.ActorCritic import ActorCritic
from utils.Diverse import init_weights, default_actor_critic
from utils.Statistics import Statistics

default_ppo_config = {
    #
    # BASIC CONFIG
    #

    # ENV CONFIG
    'env_id': 'CartPole-v0',
    'env_config': {},
    'num_steps': 128,
    'gamma': 0.99,
    'seed': 0,
    'num_worker': multiprocessing.cpu_count(),

    # NN CONFIG
    'optimizer': optim.Adam,  # if you need more control, you can define a lambda
    'lr': 5e-5,
    'clip_grad_norm': 0.5,
    # devices
    'training_device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # the device uses by the worker to compute actions they will perform
    'inference_device': torch.device('cpu'),

    # STATISTICS CONFIG
    # for more control about how to handle the stats, modify this
    'statistics': Statistics,
    # WandB init configs: https://docs.wandb.ai/ref/init
    'WandB_project': None,
    'WandB_entity': None,
    'WandB_group': None,
    'WandB_job_type': None,
    'WandB_tags': None,
    'WandB_notes': None,
    # define the frequency WandB logs gradients from model
    # don't set too low or you will reach the api limits
    'WandB_model_log_frequency': 300,

    # Actor Critic specific config
    'distribution': Categorical,
    # Lamdba-GAE: https://arxiv.org/abs/1506.02438
    'use_gae': True,
    'lambda_gae': 0.95,
    # only valuable with large batches
    'normalize_advantage': True,
    'policy_coeff': 1.0,
    'vf_coeff': 1.0,
    'entropy_coeff': 0.001,
    # neural network
    # define the template needs to inherit from AbstractActorCritic
    # can be change if you need very specific behavior
    'nn_template': ActorCritic,
    # kwargs depend on your template
    'nn_kwargs': {
        # by default no common layers (input is also the output)
        'common_layers': lambda input_size: lambda x: x,

        # keep the lambda, it's important !
        'critic_layers': lambda input_size: nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ),
        # keep the lambda, it's important !
        'actor_layers': lambda input_size, output_size: nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=1)
        ),
    },

    'common_layers_initialization': None,
    'critic_layers_initialization': lambda x: init_weights(x, function_output=lambda layer: default_actor_critic(layer, gain=1.00)),
    'actor_layers_initialization': lambda x: init_weights(x, function_output=lambda layer: default_actor_critic(layer, gain=0.01)),

    # PPO Specific config
    'ppo_epochs': 10,
    'policy_clipping_param': 0.1,
    'vf_clipping_param': None,
    'mini_batch_size': 256,
    'min_reward': -20,
    'max_reward': 20,
    'target_kl_div': 0.03,

    # ADVANCED CONFIG (don't touch if you don't know what you're doing)
    # VecEnv option, don't touch if not needed
    # see the doc for more info: https://github.com/openai/gym/blob/master/gym/vector/async_vector_env.py#L25
    'env_copy': False,
    'shared_memory': False,
    # Pytorch thread option
    'torch_num_threads': psutil.cpu_count(logical=False),
    # prevent cuDnn from doing benchmark and force him to use reproducible algorithm if possible
    # can impact the performance, don't activate if needs full performances
    # Or try to see if it impact badly your perf first
    'cudnn_trade_perf_for_reproducibility': False,
}
