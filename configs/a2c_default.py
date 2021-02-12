import multiprocessing
import psutil
import torch
from torch import optim, nn
from torch.distributions import Categorical

from models.ActorCritic import ActorCritic
from utils.Diverse import init_weights, default_actor_critic
from utils.Statistics import Statistics

default_a2c_config = {
    #
    # BASIC CONFIG
    #

    # ENV CONFIG
    'env_id': 'LunarLander-v2',
    'num_steps': 32,
    'gamma': 0.99,
    'seed': 0,
    'num_worker': multiprocessing.cpu_count() * 4,

    # NN CONFIG
    'optimizer': optim.Adam,  # if you need more control, you can define a lambda
    'lr_initial': 1e-4,
    # None means constant
    # if you want to use Pytorch scheduler, you can! It's even recommended
    # Example: 'lr_scheduler': lambda x: torch.optim.lr_scheduler.LambdaLR(x, lr_lambda=lambda epoch: 0.999**epoch),
    # x represent the optimizer, keep this structure
    'lr_scheduler': None,
    'clip_grad_norm': 0.5,
    # devices
    'training_device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # the device uses by the worker to compute actions they will perform
    'inference_device': torch.device('cpu'),

    # STATISTICS CONFIG
    # for more control about how to handle the stats, modify this
    'statistics': Statistics,
    # for smoothing you need to use SmoothedStatistics, otherwise you can use Statistics
    'metric_smoothing': 1000,
    # WandB init configs: https://docs.wandb.ai/ref/init
    'WandB_project': None,
    'WandB_entity': None,
    'WandB_group': None,
    'WandB_job_type': None,
    'WandB_tags': None,
    'WandB_notes': None,
    # define the frequency WandB logs gradients from model
    # don't set too low or you will reach the api limits
    'WandB_model_log_frequency': 100,

    # Actor Critic specific config
    'distribution': Categorical,
    # Lamdba-GAE: https://arxiv.org/abs/1506.02438
    'use_gae': True,
    'lambda_gae': 0.95,
    # only valuable with large batches
    'normalize_advantage': False,
    'vf_coeff': 1.0,
    'entropy_coeff': 1e-3,
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
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ),
        # keep the lambda, it's important !
        'actor_layers': lambda input_size, output_size: nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Softmax(dim=1)
        ),
    },

    'critic_layers_initialization': lambda x: init_weights(x,
                                                           function_output=lambda x: default_actor_critic(x, gain=1.00)),
    'actor_layers_initialization': lambda x: init_weights(x,
                                                           function_output=lambda x: default_actor_critic(x, gain=0.01)),


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

