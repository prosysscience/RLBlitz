import time

import wandb
import torch

from configs.a2c_default import default_a2c_config
from utils.Memory import Memory
from utils.Diverse import init_and_seed
from utils.vec_env.util import create_subproc_env

import torch.optim.lr_scheduler


class A2C:

    def __init__(self, config=default_a2c_config):
        wandb.init(config=config, monitor_gym=True)
        init_and_seed(config)
        self.config = config
        self.envs, self.env_info = create_subproc_env(config['env_id'], config['seed'], config['num_worker'],
                                                      config['shared_memory'], config['env_copy'], False)
        self.training_device = config['training_device']
        self.inference_device = config['inference_device']
        self.num_worker = config['num_worker']
        self.num_steps = config['num_steps']
        self.gamma = config['gamma']
        self.lambda_gae = config['lambda_gae']
        self.device = torch.device("cuda:0" if config['use_gpu'] else "cpu")
        self.statistics = config['statistics'](self.config)
        self.state_dim = self.env_info.observation_space.shape[0]
        self.action_dim = self.env_info.action_space.n
        self.lr = config['lr_initial']

        self.distribution = config['distribution']

        self.states = self.envs.reset()
        self.states_tensor = torch.from_numpy(self.states).to(self.inference_device, non_blocking=True)
        self.memory = Memory(config, self.state_dim, self.training_device, self.config['use_gae'])
        neural_network_architecture = self.config['nn_template']
        self.inference_model = neural_network_architecture(self.state_dim, self.action_dim, **config['nn_kwargs'])
        self.inference_model.to(self.inference_device, non_blocking=True)
        if self.training_device != self.inference_device:
            self.training_model = neural_network_architecture(self.state_dim, self.action_dim, **config['nn_kwargs'])
            self.training_model.load_state_dict(self.inference_model.state_dict())
            self.training_model.to(self.training_device, non_blocking=True)
        else:
            self.training_model = self.inference_model
        wandb.watch(self.training_model)
        self.optimizer = config['optimizer'](self.training_model.parameters(), lr=config['lr_initial'])
        self.scheduler = config['lr_scheduler'](self.optimizer)

    def act(self):
        self.statistics.start_act()
        self.memory.clear()
        for step_nb in range(self.num_steps):
            self.statistics.start_step()
            probabilities, values = self.inference_model(self.states_tensor)
            wandb.log({'inference_time': time.time() - self.statistics.time_start_step},
                      step=self.statistics.iteration)
            dist = self.distribution(probabilities)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            actions = actions.to('cpu', non_blocking=True)

            self.memory.actions[step_nb] = actions.to(self.training_device, non_blocking=True)
            self.memory.values[step_nb] = values.to(self.training_device, non_blocking=True)
            self.memory.logprobs[step_nb] = logprobs.to(self.training_device, non_blocking=True)
            self.memory.entropy += dist.entropy().mean()

            next_states, rewards, dones, _ = self.envs.step(actions.numpy())

            rewards_tensor = torch.from_numpy(rewards).to(self.training_device, non_blocking=True)
            dones_tensor = torch.from_numpy(dones).to(self.training_device, non_blocking=True)

            self.states = next_states
            self.states_tensor = torch.from_numpy(self.states).to(self.inference_device, non_blocking=True)

            self.memory.rewards[step_nb] = rewards_tensor
            self.memory.is_terminals[step_nb] = dones_tensor

            self.statistics.add_rewards(rewards)
            # Statistics
            for worker_id, done in enumerate(dones):
                if done:
                    wandb.log({'episode_return': self.statistics.episode_return[worker_id],
                               'episode_len': self.statistics.episode_len[worker_id],
                               'episode_number': self.statistics.episode_number}, step=self.statistics.iteration)
                    self.statistics.episode_done(worker_id)

            self.statistics.end_step()
        self.statistics.end_act()

    def update(self):
        self.statistics.start_update()
        self.optimizer.zero_grad(set_to_none=True)

        advantage = self._compute_advantage(self.config['use_gae'])

        critic_loss = advantage.pow(2).mean()

        if self.config['normalize_advantage']:
            with torch.no_grad():
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(self.memory.logprobs * advantage.detach()).mean()

        loss = actor_loss + \
               self.config['vf_coeff'] * critic_loss - \
               self.config['entropy_coeff'] * self.memory.entropy

        wandb.log({'total_loss': loss, 'actor_loss': actor_loss, 'critic_loss': critic_loss,
                   'entropy': self.memory.entropy, 'lr': self.scheduler.get_last_lr()[-1]},
                  step=self.statistics.iteration)

        loss.backward()
        if self.config['clip_grad_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), self.config['clip_grad_norm'])
        self.optimizer.step()
        self.scheduler.step()
        self.states_tensor = self.states_tensor.to(self.inference_device, non_blocking=True)
        self.inference_model.load_state_dict(self.training_model.state_dict())
        self.statistics.end_update()
        wandb.log({'training_time': time.time() - self.statistics.time_start_update},
                  step=self.statistics.iteration)

    def _compute_advantage(self, use_gae=True):
        if use_gae:
            with torch.no_grad():
                self.states_tensor = self.states_tensor.to(self.training_device, non_blocking=True)
                returns = torch.empty((len(self.memory), self.num_worker), dtype=torch.float,
                                      device=self.training_device)
                not_terminal = torch.logical_not(self.memory.is_terminals)
                self.memory.values[self.num_steps] = self.training_model.critic_only(self.states_tensor)
                values = self.memory.values.view(self.memory.values.shape[0], self.memory.values.shape[1])
                gae = 0
                index = len(self.memory) - 1
                for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
                    delta = reward + ((non_terminal * self.gamma * values[index + 1]) - values[index])
                    gae = delta + self.gamma * self.lambda_gae * non_terminal * gae
                    returns[index] = gae
                    index -= 1
            # view is better than squeeze because it avoid copy
            advantage = returns - values[:-1]
        else:
            with torch.no_grad():
                self.states_tensor = self.states_tensor.to(self.training_device, non_blocking=True)
                returns = torch.empty((len(self.memory), self.num_worker), dtype=torch.float,
                                      device=self.training_device)
                not_terminal = torch.logical_not(self.memory.is_terminals)
                return_value = self.training_model.critic_only(self.states_tensor).view(-1)
                index = len(self.memory) - 1
                for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
                    return_value = reward + (non_terminal * self.gamma * return_value)
                    returns[index] = return_value
                    index -= 1
            # view is better than squeeze because it avoid copy
            values = self.memory.values.view(self.memory.values.shape[0], self.memory.values.shape[1])
            advantage = returns - values
        return advantage

    def train(self):
        self.statistics.start_train()
        self.statistics.episode_this_iter = 0
        self.act()
        self.update()
        self.statistics.end_train()
        wandb.log({'iteration': self.statistics.iteration,
                   'episode_this_iter': self.statistics.episode_this_iter,
                   'total_steps': self.statistics.total_step,
                   'time_this_iter': time.time() - self.statistics.time_start_train},
                  step=self.statistics.iteration)

    def render(self, number_worker=None):
        if number_worker is None:
            number_worker = self.num_worker
        rendering_env, _ = create_subproc_env(self.config['env_id'], self.config['seed'], number_worker,
                                              self.config['shared_memory'], self.config['env_copy'], True)
        rendering_states = rendering_env.reset()
        with torch.no_grad():
            episode_done_worker = torch.zeros(number_worker, dtype=torch.bool)
            while torch.sum(episode_done_worker) < number_worker:
                rendering_env.render(mode='rgb_array')
                states_tensor = torch.from_numpy(rendering_states)
                probabilities = self.inference_model.actor_only(states_tensor)
                dist = self.distribution(probabilities)
                actions = dist.sample()
                actions = actions.to('cpu', non_blocking=True)
                rendering_states, rewards, dones, _ = rendering_env.step(actions.numpy())
                episode_done_worker += dones
            rendering_env.render(mode='rgb_array')
        rendering_env.close()

    def save_model(self, path='a2c_default'):
        torch.save(self.inference_model.state_dict(), path + ".h5")
        wandb.save(path + '.h5')

    def load_model(self, path='a2c_default'):
        wandb.unwatch(self.training_model)
        self.inference_model = torch.load(path).to(self.inference_device)
        self.training_model = torch.load(path).to(self.training_device)
        wandb.watch(self.training_model)

    def save_agent_checkpoint(self, path='a2c_default_checkpoint'):
        pass
