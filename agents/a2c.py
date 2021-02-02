import wandb
import torch

from configs.a2c_default import default_a2c_config
from models.ActorCritic import ActorCritic
from utils.Memory import Memory
from utils.InitLibrary import init_and_seed
from utils.Statistics import Statistics
from utils.vec_env.util import create_subproc_env

import torch.optim.lr_scheduler


class A2C:

    def __init__(self, config=default_a2c_config):
        wandb.init(config=config, monitor_gym=True)
        init_and_seed(config)
        self.config = config
        self.envs, self.env_info = create_subproc_env(config['env_id'], config['seed'], config['num_worker'],
                                                      config['shared_memory'], config['env_copy'])
        self.num_worker = config['num_worker']
        self.num_steps = config['num_steps']
        self.gamma = config['gamma']
        self.device = torch.device("cuda:0" if config['use_gpu'] else "cpu")
        self.statistics = Statistics(self.num_worker)
        self.state_dim = self.env_info.observation_space.shape[0]
        self.action_dim = self.env_info.action_space.n
        self.memory = Memory(config, self.state_dim)
        self.lr = config['lr_initial']
        self.model = ActorCritic(self.state_dim, self.action_dim, config['activation_fn'],
                                 config['hidden_size'], config['logistic_function'])
        wandb.watch(self.model)
        self.distribution = config['distribution']
        self.optimizer = config['optimizer'](self.model.parameters(), lr=config['lr_initial'])
        self.scheduler = config['lr_scheduler'](self.optimizer)
        self.states = self.envs.reset()

    def act(self):
        self.memory.clear()
        for step_nb in range(self.num_steps):
            states_tensor = torch.from_numpy(self.states)
            probabilities, values = self.model(states_tensor)
            dist = self.distribution(probabilities)
            actions = dist.sample()
            actions = actions.to('cpu', non_blocking=True)
            logprobs = dist.log_prob(actions)

            self.memory.actions[step_nb] = actions
            self.memory.values[step_nb] = values
            self.memory.logprobs[step_nb] = logprobs
            self.memory.entropy += dist.entropy().mean()

            next_states, rewards, dones, _ = self.envs.step(actions.numpy())

            rewards_tensor = torch.from_numpy(rewards)
            dones_tensor = torch.from_numpy(dones)
            self.memory.rewards[step_nb] = rewards_tensor
            self.memory.is_terminals[step_nb] = dones_tensor

            self.states = next_states

            self.statistics.episode_return += rewards
            # Statistics
            for worker_id, done in enumerate(dones):
                if done:
                    wandb.log({'episode_number': self.statistics.episode_number,
                               'episode_return': self.statistics.episode_return[worker_id]})
                    self.statistics.episode_return[worker_id] = 0
                    self.statistics.episode_number += 1

            self.statistics.total_step += self.num_worker

    def update(self):
        with torch.no_grad():
            returns = torch.empty((len(self.memory), self.config['num_worker']), dtype=torch.float)
            not_terminal = torch.logical_not(self.memory.is_terminals)
            states_tensor = torch.from_numpy(self.states)
            return_value = self.model.value_network(states_tensor).view(-1)
            index = len(self.memory) - 1
            for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
                return_value = reward + (non_terminal * self.gamma * return_value)
                returns[index] = return_value
                index -= 1

        # view is better than squeeze because it
        values = self.memory.values.view(self.memory.values.shape[0], self.memory.values.shape[1])

        advantage = returns - values

        critic_loss = advantage.pow(2).mean()

        if self.config['normalize_advantage']:
            with torch.no_grad():
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(self.memory.logprobs * advantage.detach()).mean()


        loss = actor_loss + self.config['vf_coeff'] * critic_loss - self.config['entropy_coeff'] * self.memory.entropy

        wandb.log({'total_loss': loss, 'actor_loss': actor_loss, 'critic_loss': critic_loss,
                   'entropy': self.memory.entropy, 'lr': self.scheduler.get_last_lr()[-1]})

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config['clip_grad_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
        self.optimizer.step()
        self.scheduler.step()

    def train(self):
        self.act()
        self.update()
        self.statistics.iteration += 1
        wandb.log({'iteration': self.statistics.iteration})

