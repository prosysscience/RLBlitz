from gym.vector import AsyncVectorEnv, SyncVectorEnv
from gym.vector.tests.utils import make_env


def create_subproc_env(env_id='LunarLander-v2', seed=0, num_worker=1, shared_memory=False, copy=False):
    env_fn = [make_env(env_id, seed) for i in range(num_worker)]
    return AsyncVectorEnv(env_fn, shared_memory=shared_memory, copy=copy), make_env(env_id, seed)()
