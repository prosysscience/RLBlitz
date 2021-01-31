from gym.vector.tests.utils import make_env

from utils.vec_env.multiprocessing_env import SubprocVecEnv


def create_subproc_env(config):
    env_fn = [make_env(config['env_id'], config['seed']) for i in range(config['num_worker'])]
    return SubprocVecEnv(env_fn)