from gym.vector import AsyncVectorEnv
from gym.vector.tests.utils import make_env


def create_subproc_env(config):
    env_fn = [make_env(config['env_id'], config['seed']) for i in range(config['num_worker'])]
    return AsyncVectorEnv(env_fn, shared_memory=config['shared_memory'], copy=config['env_copy'])