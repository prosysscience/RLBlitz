import copy
import os

import cloudpickle
import wandb
import torch

from agents.abstract_agent import AbstractAgent
from configs.a2c_default import default_a2c_config
from utils.Memory import Memory
from utils.Diverse import init_and_seed, MockLRScheduler, parse_scheduler
from utils.ParameterScheduler import ConstantScheduler, ParameterScheduler
from utils.vec_env.util import create_subproc_env


class A2C(AbstractAgent):

    def __init__(self, config=default_a2c_config):
        super().__init__(config)
        wandb.init(config=config, monitor_gym=True, resume="allow", project=config['WandB_project'],
                   entity=config['WandB_entity'], group=config['WandB_group'], job_type=config['WandB_job_type'],
                   tags=config['WandB_tags'], notes=config['WandB_notes'])
        init_and_seed(config)
        self.scheduler_parameters = []
        self.config = config
        self.envs, self.env_info = create_subproc_env(config['env_id'], config['env_config'], config['seed'], config['num_worker'],
                                                      config['shared_memory'], config['env_copy'], False)
        self.training_device = config['training_device']
        self.inference_device = config['inference_device']
        self.num_worker = config['num_worker']
        self.num_steps = config['num_steps']
        self.gamma = parse_scheduler(config['gamma'])
        self.lambda_gae = parse_scheduler(config['lambda_gae'])
        self.scheduler_parameters.append(self.gamma)
        self.scheduler_parameters.append(self.lambda_gae)
        self.statistics = config['statistics'](self.config)
        self.state_dim = self.env_info.observation_space.shape[0]
        self.action_dim = self.env_info.action_space.n
        self.lr = parse_scheduler(config['lr'])
        self.policy_coeff = parse_scheduler(config['policy_coeff'])
        self.vf_coeff = parse_scheduler(config['vf_coeff'])
        self.entropy_coeff = parse_scheduler(config['entropy_coeff'])
        self.scheduler_parameters.append(self.policy_coeff)
        self.scheduler_parameters.append(self.vf_coeff)
        self.scheduler_parameters.append(self.entropy_coeff)

        self.distribution = config['distribution']

        self.clip_grad_norm = parse_scheduler(self.config['clip_grad_norm'])
        self.scheduler_parameters.append(self.clip_grad_norm)

        self.states = self.envs.reset()
        self.states_tensor = torch.from_numpy(self.states).to(self.inference_device, non_blocking=True)
        self.memory = Memory(config, self.state_dim, self.training_device, self.config['use_gae'])
        neural_network_architecture = self.config['nn_template']
        self.inference_model = neural_network_architecture(self.state_dim, self.action_dim, **config['nn_kwargs'])
        if self.config['critic_layers_initialization'] is not None:
            self.config['critic_layers_initialization'](self.inference_model.get_critic())
            self.config['actor_layers_initialization'](self.inference_model.get_actor())
        self.inference_model.to(self.inference_device, non_blocking=True)
        if self.training_device != self.inference_device:
            self.training_model = neural_network_architecture(self.state_dim, self.action_dim, **config['nn_kwargs'])
            self.training_model.load_state_dict(self.inference_model.state_dict())
            self.training_model.to(self.training_device, non_blocking=True)
        else:
            self.training_model = self.inference_model
        wandb.watch(self.training_model, log_freq=self.config['WandB_model_log_frequency'])
        self.optimizer = config['optimizer'](self.training_model.parameters(), lr=config['lr_initial'])

    def act(self):
        wandb.log(self.statistics.start_act(), step=self.statistics.get_iteration())
        self.memory.clear()
        for step_nb in range(self.num_steps):
            wandb.log(self.statistics.start_step(), step=self.statistics.get_iteration())
            wandb.log(self.statistics.start_inference(), step=self.statistics.get_iteration())
            probabilities, values = self.inference_model(self.states_tensor)
            wandb.log(self.statistics.end_inference(), step=self.statistics.get_iteration())
            dist = self.distribution(probabilities)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            actions = actions.to('cpu', non_blocking=True)

            self.memory.actions[step_nb] = actions.to(self.training_device, non_blocking=True)
            self.memory.values[step_nb] = values.to(self.training_device, non_blocking=True)
            self.memory.logprobs[step_nb] = log_prob.to(self.training_device, non_blocking=True)
            self.memory.entropy += dist.entropy().mean()

            wandb.log(self.statistics.start_env_wait(), step=self.statistics.get_iteration())
            next_states, rewards, dones, _ = self.envs.step(actions.numpy())
            wandb.log(self.statistics.end_env_wait(), step=self.statistics.get_iteration())

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
                    wandb.log(self.statistics.episode_done(worker_id), step=self.statistics.get_iteration())
            wandb.log(self.statistics.end_step(), step=self.statistics.get_iteration())
        self.increment_scheduler(self.statistics.get_episodes_this_iter(), criteria='episode')
        wandb.log(self.statistics.end_act(), step=self.statistics.get_iteration())

    def update(self):
        wandb.log(self.statistics.start_update(), step=self.statistics.get_iteration())
        self.optimizer.zero_grad(set_to_none=True)

        computed_return = self._compute_return(self.config['use_gae'])
        values = self.memory.values.view(self.memory.values.shape[0], self.memory.values.shape[1])
        if self.config['use_gae']:
            advantage = computed_return - values[:-1]
        else:
            advantage = computed_return - values
        critic_loss = advantage.pow(2).mean()

        if self.config['normalize_advantage']:
            with torch.no_grad():
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(self.memory.logprobs * advantage.detach()).mean()

        # we average entropy over numb of steps
        self.memory.entropy /= self.num_steps

        loss = self.policy_coeff.get_current_value() * actor_loss + self.vf_coeff.get_current_value() * critic_loss - self.entropy_coeff.get_current_value() * self.memory.entropy

        wandb.log({'Algorithm/total_loss': loss,
                   'Algorithm/actor_loss': actor_loss,
                   'Algorithm/critic_loss': critic_loss,
                   'Statistics/entropy': self.memory.entropy,
                   'Algorithm/LR': self.lr.get_current_value()},
                  step=self.statistics.get_iteration_nb())

        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), self.clip_grad_norm.get_current_value())
        self.optimizer.step()
        self.states_tensor = self.states_tensor.to(self.inference_device, non_blocking=True)
        self.inference_model.load_state_dict(self.training_model.state_dict())
        wandb.log(self.statistics.end_update(), step=self.statistics.get_iteration_nb())

    def _compute_return(self, use_gae=True):
        with torch.no_grad():
            self.states_tensor = self.states_tensor.to(self.training_device, non_blocking=True)
            returns = torch.empty((len(self.memory), self.num_worker), dtype=torch.float,
                                  device=self.training_device)
            not_terminal = torch.logical_not(self.memory.is_terminals)
            if use_gae:
                self.memory.values[self.num_steps] = self.training_model.critic_only(self.states_tensor)
                values = self.memory.values.view(self.memory.values.shape[0], self.memory.values.shape[1])
                gae = 0
                index = len(self.memory) - 1
                for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
                    delta = reward + ((non_terminal * self.gamma.get_current_value() * values[index + 1]) - values[index])
                    gae = delta + self.gamma .get_current_value() * self.lambda_gae.get_current_value() * non_terminal * gae
                    returns[index] = gae
                    index -= 1
            else:
                return_value = self.training_model.critic_only(self.states_tensor).view(-1)
                index = len(self.memory) - 1
                for reward, non_terminal in zip(reversed(self.memory.rewards), reversed(not_terminal)):
                    return_value = reward + (non_terminal * self.gamma .get_current_value() * return_value)
                    returns[index] = return_value
                    index -= 1
        return returns

    def train(self):
        wandb.log(self.statistics.start_train(), step=self.statistics.get_iteration())
        self.act()
        self.update()
        self.increment_scheduler(1, criteria='train_iter')
        wandb.log(self.statistics.end_train(),
                  step=self.statistics.get_iteration_nb())

    def render(self, number_workers=None, mode='human'):
        if number_workers is None:
            number_workers = self.num_worker
        render_config = copy.deepcopy(self.config)
        render_config['num_worker'] = number_workers
        rendering_statistics = self.config['statistics'](render_config)
        rendering_env, _ = create_subproc_env(self.config['env_id'], self.config['seed'], number_workers,
                                              self.config['shared_memory'], self.config['env_copy'], True)
        rendering_states = rendering_env.reset()
        with torch.no_grad():
            episode_done_worker = torch.zeros(number_workers, dtype=torch.bool)
            while torch.sum(episode_done_worker) < number_workers:
                rendering_env.get_images()
                states_tensor = torch.from_numpy(rendering_states).to(self.inference_device, non_blocking=True)
                probabilities = self.inference_model.actor_only(states_tensor)
                dist = self.distribution(probabilities)
                actions = dist.sample()
                actions = actions.to('cpu', non_blocking=True)
                rendering_states, rewards, dones, _ = rendering_env.step(actions.numpy())
                episode_done_worker += dones
                rendering_statistics.add_rewards(rewards)
                # Statistics
                for worker_id, done in enumerate(dones):
                    if done:
                        rendering_statistics.episode_done(worker_id)
            rendering_env.get_images()
        rendering_env.close()
        return rendering_statistics

    def save_model(self, filename='a2c_default.h5'):
        torch.save(self.inference_model.state_dict(), filename)
        wandb.save(filename)

    def load_model(self, filename='a2c_default'):
        wandb.unwatch(self.training_model)
        self.inference_model = torch.load(filename).to(self.inference_device)
        self.training_model = torch.load(filename).to(self.training_device)
        wandb.watch(self.training_model, log_freq=self.config['WandB_model_log_frequency'])

    def save_agent_checkpoint(self, filename='a2c_default_checkpoint.pickle'):
        to_save = {'statistics': self.statistics,
                   'inference_model': self.inference_model.state_dict(),
                   'training_model': self.training_model.state_dict(),
                   'optimizer': self.optimizer.state_dict(),
                   'num_worker': self.num_worker,
                   'num_steps': self.num_steps,
                   'lr': self.lr,
                   'policy_coeff': self.policy_coeff,
                   'vf_coeff': self.vf_coeff,
                   'gamma': self.gamma,
                   'lambda_gae': self.lambda_gae,
                   'entropy_coeff': self.entropy_coeff,
                   'clip_grad_norm': self.clip_grad_norm,
                   'memory': self.memory,
                   'distribution': self.distribution,
                   'WandB_id': wandb.util.generate_id(),
                   'config': self.config}
        with open(filename, 'wb') as f:
            cloudpickle.dump(to_save, f)

    @classmethod
    def load_agent_checkpoint(cls, filename='a2c_default_checkpoint.pickle'):
        with open(filename, 'rb') as f:
            saved_data = cloudpickle.load(f)
            os.environ["WANDB_RUN_ID"] = saved_data['WandB_id']
            new_class = cls(saved_data['config'])
            new_class.memory = saved_data['memory']
            new_class.statistics = saved_data['statistics']
            new_class.distribution = saved_data['distribution']
            new_class.inference_model.load_state_dict(saved_data['inference_model'])
            new_class.training_model.load_state_dict(saved_data['training_model'])
            new_class.optimizer.load_state_dict(saved_data['optimizer'])
            new_class.num_worker = saved_data['num_worker']
            new_class.num_steps = saved_data['num_steps']
            new_class.lr = saved_data['lr']
            new_class.policy_coeff = saved_data['policy_coeff']
            new_class.vf_coeff = saved_data['vf_coeff']
            new_class.gamma = saved_data['gamma']
            new_class.lambda_gae = saved_data['lambda_gae']
            new_class.entropy_coeff = saved_data['entropy_coeff']
            new_class.clip_grad_norm = saved_data['clip_grad_norm']
            return new_class

    def increment_scheduler(self, steps, criteria='episode'):
        for scheduler in self.scheduler_parameters:
            if scheduler.criteria == criteria:
                scheduler.inc_step(steps)
        if isinstance(self.lr, ParameterScheduler) and not isinstance(self.lr, ConstantScheduler):
            if self.lr.criteria == criteria:
                self.lr.inc_step(steps)
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr.get_current_value()




