import torch
import multiprocessing
import psutil
from torch import optim, nn

from utils.Statistics import Statistics, SmoothedStatistics

default_config = {
    #
    # BASIC CONFIG
    #

    # ENV CONFIG
    'env_id': 'LunarLander-v2',
    'num_steps': 5,
    'gamma': 0.99,
    'seed': 0,
    'num_worker': multiprocessing.cpu_count() * 8,

    # NN CONFIG
    # if nn_architecture is a list, it represents the size of the hidden layers in each NN
    # if you need special architecture, you can define your own neural network extending the AbstractActorCritic class
    # and pass it simply, example: 'nn_architecture': MyNetwork,
    # you can define extra key in the config to pass more info to your neural network
    'nn_architecture': [128, 128],
    'activation_fn': nn.ReLU(),
    'optimizer': optim.Adam,  # if you need more control, you can define a lambda
    'lr_initial': 1e-4,
    # None means constant
    # if you want to use Pytorch scheduler, you can! It's even recommended
    # Example: 'lr_scheduler': lambda x: torch.optim.lr_scheduler.LambdaLR(x, lr_lambda=lambda epoch: 0.999**epoch),
    # x represent the optimizer, keep this structure
    'lr_scheduler': None,
    'clip_grad_norm': None,
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