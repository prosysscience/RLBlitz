import numpy as np
import torch

from configs.a2c_default import default_a2c_config
from models.ActorCritic import ActorCritic
from utils.Memory import Memory
from utils.subproc_vec_env import create_subproc_env


class A2C:

    def __init__(self, config=default_a2c_config):
        self.envs = create_subproc_env(config['env_id'], config['num_worker'], config['seed'])
        self.num_steps = config['num_steps']
        self.gamma = config['gamma']
        self.device = torch.device("cuda:0" if config['use_gpu'] else "cpu")
        self.memory = Memory()
        self.state_dim = self.envs.observation_space.shape[0]
        self.action_dim = self.envs.action_space.n
        self.lr_scheduler = config['lr']
        self.model = ActorCritic(self.state_dim, self.action_dim, config['hidden_size'])
        self.optimizer = config['optimizer'](self.model.parameters())
        self.states = self.envs.reset()

    def act(self):
        self.memory.clear()
        for _ in range(self.num_steps):
            states_tensor = torch.from_numpy(self.states)
            distribution, value = self.model(states_tensor)

    def update(self):
        rewards = np.empty((len(self.memory), ), dtype=np.float)
        not_terminal = 1 - self.memory.is_terminals
        discounted_reward = 0
        index = len(self.memory) - 1
        for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
            discounted_reward = reward + (non_terminal * self.gamma * discounted_reward)
            rewards[index] = discounted_reward
            index -= 1
