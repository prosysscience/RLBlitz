import torch
import multiprocessing
import psutil

default_config = {
    # BASIC CONFIG
    'env_id': 'LunarLander-v2',
    'seed': 0,
    'num_worker': multiprocessing.cpu_count(),
    #TODO
    'use_gpu': torch.cuda.is_available(),
    # we inference is done on CPU and the training can be done on GPU is activated
    'workers_use_cpu': True,

    # ADVANCED CONFIG (don't touch if you don't know what you're doing)
    # VecEnv option, don't touch if not needed
    # see the doc for more info: https://github.com/openai/gym/blob/master/gym/vector/async_vector_env.py#L25
    'env_copy': False,
    'shared_memory': False,
    # Pytorch thread option
    'torch_num_threads': psutil.cpu_count(logical=False),
}