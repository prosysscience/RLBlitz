import torch
import multiprocessing
import psutil
from torch import optim, nn

from utils.Statistics import Statistics

default_config = {
    #
    # BASIC CONFIG
    #

    # ENV CONFIG
    'env_id': 'LunarLander-v2',
    'num_steps': 5,
    'gamma': 0.99,
    'seed': 0,
    'num_worker': multiprocessing.cpu_count() * 4,

    # NN CONFIG
    # if nn_architecture is a list, it represents the size of the hidden layers in each NN
    # if you need special architecture, you can define your own neural network extending the AbstractActorCritic class
    # and pass it simply, example: 'nn_architecture': MyNetwork,
    # you can define extra key in the config to pass more info to your neural network
    'nn_architecture': [128, 128],
    'activation_fn': nn.ReLU(),
    'optimizer': optim.Adam,  # if you need more control, you can define a lambda
    'lr_initial': 1e-4,
    # default scheduler is constant, x represent the optimizer
    'lr_scheduler': lambda x: torch.optim.lr_scheduler.LambdaLR(x, lr_lambda=lambda epoch: 1.0),
    'clip_grad_norm': None,
    'use_gpu': torch.cuda.is_available(),
    # we inference is done on CPU and the training can be done on GPU is activated
    'workers_use_gpu': False,

    # STATISTICS CONFIG
    # for more control about how to handle the stats, modify this
    'statistics': Statistics,

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