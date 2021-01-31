import numpy as np
import torch

from configs.a2c_default import default_a2c_config
from models.ActorCritic import ActorCritic
from utils.Memory import Memory
from utils.vec_env.util import create_subproc_env


class A2C:

    def __init__(self, config=default_a2c_config):
        self.envs = create_subproc_env(config)
        self.num_steps = config['num_steps']
        self.gamma = config['gamma']
        self.device = torch.device("cuda:0" if config['use_gpu'] else "cpu")
        self.memory = Memory()
        self.state_dim = self.envs.observation_space.shape[0]
        self.action_dim = self.envs.action_space[0].n
        self.lr_scheduler = config['lr']
        self.model = ActorCritic(self.state_dim, self.action_dim, config['activation_fn'], config['hidden_size'])
        self.optimizer = config['optimizer'](self.model.parameters())
        self.states = self.envs.reset()

    def act(self):
        self.memory.clear()
        for _ in range(self.num_steps):
            states_tensor = torch.from_numpy(self.states)
            distributions, values = self.model(states_tensor)
            actions = distributions.sample()

            self.memory.values.append(values)
            self.memory.logprobs.append(distributions.log_prob(actions))
            self.memory.entropy += distributions.entropy().mean()

            next_states, rewards, dones, _ = self.envs.step(actions.cpu().numpy())

            self.memory.rewards.append(torch.from_numpy(rewards).unsqueeze(1))
            self.memory.is_terminals.append(torch.from_numpy(dones).unsqueeze(1))

            self.states = next_states

    def update(self):
        rewards = np.empty((len(self.memory), ), dtype=np.float)
        not_terminal = 1 - self.memory.is_terminals
        discounted_reward = 0
        index = len(self.memory) - 1
        for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
            discounted_reward = reward + (non_terminal * self.gamma * discounted_reward)
            rewards[index] = discounted_reward
            index -= 1
