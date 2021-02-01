import numpy as np
import torch

from configs.a2c_default import default_a2c_config
from models.ActorCritic import ActorCritic
from utils.Memory import Memory
from utils.vec_env.util import create_subproc_env

import torch.optim.lr_scheduler


class A2C:

    def __init__(self, config=default_a2c_config):
        self.envs = create_subproc_env(config)
        self.num_worker = config['num_worker']
        self.num_steps = config['num_steps']
        self.gamma = config['gamma']
        self.device = torch.device("cuda:0" if config['use_gpu'] else "cpu")
        self.memory = Memory()
        self.state_dim = self.envs.observation_space.shape[0]
        self.action_dim = self.envs.action_space[0].n
        self.lr = config['lr_initial']
        self.model = ActorCritic(self.state_dim, self.action_dim, config['activation_fn'], config['hidden_size'])
        self.optimizer = config['optimizer'](self.model.parameters(), lr=self.lr)
        self.scheduler = config['lr_scheduler'](self.optimizer)
        self.states = self.envs.reset()
        self.step_nb = 0

    def act(self):
        self.memory.clear()
        for _ in range(self.num_steps):
            states_tensor = torch.from_numpy(self.states)
            distributions, values = self.model(states_tensor)
            actions = distributions.sample()

            self.memory.actions.append(actions)
            self.memory.values.append(values)
            self.memory.logprobs.append(distributions.log_prob(actions))
            self.memory.entropy += distributions.entropy().mean()

            next_states, rewards, dones, _ = self.envs.step(actions.cpu().numpy())

            self.memory.rewards.append(torch.from_numpy(rewards).unsqueeze(1))
            self.memory.is_terminals.append(torch.from_numpy(dones).unsqueeze(1))

            self.states = next_states

            self.step_nb += self.num_worker

    def update(self):
        # TODO improve, no list, directly use a tensor
        returns = []
        not_terminal = ~torch.cat(self.memory.is_terminals)
        with torch.no_grad():
            states_tensor = torch.from_numpy(self.states)
            discounted_rewards = self.model.critic(states_tensor)
        for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
            discounted_rewards = reward + (non_terminal * self.gamma * discounted_rewards)
            returns.insert(0, discounted_rewards)

        returns = torch.cat(returns)
        log_probs = torch.cat(self.memory.logprobs)
        values = torch.cat(self.memory.values)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.memory.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
