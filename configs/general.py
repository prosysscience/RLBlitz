import torch
import multiprocessing

default_config = {
    'num_worker': multiprocessing.cpu_count(),
    'use_gpu': torch.cuda.is_available(),
}