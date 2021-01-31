import torch
import multiprocessing

default_config = {
    'env_id': 'LunarLander-v2',
    'seed': 0,
    'num_worker': multiprocessing.cpu_count(),
    #TODO
    'use_gpu': torch.cuda.is_available(),
    # we inference is done on CPU and the training can be done on GPU is activated
    'workers_use_cpu': True,
}