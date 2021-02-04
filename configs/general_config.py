import torch
import multiprocessing
import psutil

from utils.Statistics import Statistics

default_config = {
    # BASIC CONFIG
    'env_id': 'LunarLander-v2',
    'num_steps': 5,
    'gamma': 0.99,
    'seed': 0,
    'num_worker': multiprocessing.cpu_count() * 4,
    #TODO
    'use_gpu': torch.cuda.is_available(),
    # we inference is done on CPU and the training can be done on GPU is activated
    'workers_use_gpu': False,
    # for more control about how to handle the stats, modify this
    'statistics': Statistics,

    # ADVANCED CONFIG (don't touch if you don't know what you're doing)
    # VecEnv option, don't touch if not needed
    # see the doc for more info: https://github.com/openai/gym/blob/master/gym/vector/async_vector_env.py#L25
    'env_copy': False,
    'shared_memory': False,
    # Pytorch thread option
    'torch_num_threads': psutil.cpu_count(logical=False),
}