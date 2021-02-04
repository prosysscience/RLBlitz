import gym
from gym.vector import AsyncVectorEnv
from gym.wrappers.monitor import Monitor

from utils.vec_env.dummy_vec_env import DummyVecEnv


def make_env(env_name, seed=0, wrapper=None):
    def _make():
        env = gym.make(env_name)
        env.seed(seed)
        if wrapper is not None:
            env = wrapper(env)
        return env
    return _make


def create_subproc_env(env_id='LunarLander-v2', seed=0, num_worker=1, shared_memory=False, copy=False, render=False):
    if render:
        env_fn = [make_env(env_id, seed + i, wrapper=lambda x: Monitor(x, directory="./tmp/videos", force=True))
                  for i in range(num_worker)]
        return DummyVecEnv(env_fn), None
    else:
        env_fn = [make_env(env_id, seed + i) for i in range(num_worker)]
        return AsyncVectorEnv(env_fn, shared_memory=shared_memory, copy=copy), make_env(env_id, seed)()