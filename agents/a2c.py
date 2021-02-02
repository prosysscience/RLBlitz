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
        self.memory = Memory()
        self.statistics = Statistics(self.num_worker)
        self.state_dim = self.env_info.observation_space.shape[0]
        self.action_dim = self.env_info.action_space.n
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
        for _ in range(self.num_steps):
            states_tensor = torch.from_numpy(self.states)
            probabilities, values = self.model(states_tensor)
            dist = self.distribution(probabilities)
            actions = dist.sample()
            actions = actions.to('cpu', non_blocking=True)
            logprobs = dist.log_prob(actions)

            self.memory.actions.append(actions)
            self.memory.values.append(values)
            self.memory.logprobs.append(logprobs)
            self.memory.entropy += dist.entropy().mean()

            next_states, rewards, dones, _ = self.envs.step(actions.numpy())

            self.memory.rewards.append(torch.from_numpy(rewards).unsqueeze(1))
            self.memory.is_terminals.append(torch.from_numpy(dones).unsqueeze(1))

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
        # TODO improve, no list, directly use a tensor
        returns = []
        with torch.no_grad():
            not_terminal = torch.logical_not(torch.cat(self.memory.is_terminals))
            states_tensor = torch.from_numpy(self.states)
            return_value = self.model.value_network(states_tensor)
            for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
                return_value = reward + (non_terminal * self.gamma * return_value)
                returns.insert(0, return_value)
            returns = torch.cat(returns).float()

        log_probs = torch.cat(self.memory.logprobs)
        values = torch.cat(self.memory.values)

        advantage = returns - values

        critic_loss = advantage.pow(2).mean()

        if self.config['normalize_advantage']:
            with torch.no_grad():
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(log_probs * advantage.detach()).mean()


        loss = actor_loss + 0.5 * critic_loss - 1e-3 * self.memory.entropy

        wandb.log({'total_loss': loss, 'actor_loss': actor_loss, 'critic_loss': critic_loss,
                   'entropy': self.memory.entropy, 'lr': self.scheduler.get_last_lr()[-1]})

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
        self.optimizer.step()
        self.scheduler.step()

    def train(self):
        self.act()
        self.update()
        self.statistics.iteration += 1
        wandb.log({'iteration': self.statistics.iteration})

