import torch
import wandb

from agents.a2c import A2C
from configs.ppo_default import default_ppo_config


class PPO(A2C):

    def __init__(self, config=default_ppo_config):
        super().__init__(config)
        self.ppo_epoch = config['ppo_epochs']
        self.policy_clipping_param = config['policy_clipping_param']
        self.mini_batch_size = config['mini_batch_size']
        self.target_kl_div = config['target_kl_div']
        self.vf_clipping_param = config['vf_clipping_param']

    def act(self):
        wandb.log(self.statistics.start_act(), step=self.statistics.get_iteration())
        self.memory.clear()
        for step_nb in range(self.num_steps):
            wandb.log(self.statistics.start_step(), step=self.statistics.get_iteration())
            wandb.log(self.statistics.start_inference(), step=self.statistics.get_iteration())
            with torch.no_grad():
                probabilities, values = self.inference_model(self.states_tensor)
                wandb.log(self.statistics.end_inference(), step=self.statistics.get_iteration())
                dist = self.distribution(probabilities)
                actions = dist.sample()
                log_prob = dist.log_prob(actions)
                actions = actions.to('cpu', non_blocking=True)

            self.memory.states[step_nb] = self.states_tensor.to(self.training_device, non_blocking=True)
            self.memory.actions[step_nb] = actions.to(self.training_device, non_blocking=True)
            self.memory.values[step_nb] = values.to(self.training_device, non_blocking=True)
            self.memory.logprobs[step_nb] = log_prob.to(self.training_device, non_blocking=True)

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
        wandb.log(self.statistics.end_act(), step=self.statistics.get_iteration())

    def update(self):
        wandb.log(self.statistics.start_update(), step=self.statistics.get_iteration())
        self.memory.rewards = torch.clamp(self.memory.rewards, self.config['min_reward'], self.config['max_reward'])
        computed_return = self._compute_return(self.config['use_gae'])
        with torch.no_grad():
            values = self.memory.values.view(self.memory.values.shape[0], self.memory.values.shape[1])
            if self.config['use_gae']:
                advantage = computed_return - values[:-1]
            else:
                advantage = computed_return - values
        accumulated_kl_div = 0
        number_epoch_done = 0
        for epoch in range(self.ppo_epoch):
            accumulated_epoch_kl_div = 0
            nb_mini_batches_done = 0
            with torch.no_grad():
                permutation = torch.randperm(self.memory.states.size(0) - 1)
            for i in range(0, self.memory.states.size(0) - 1, self.mini_batch_size):
                self.optimizer.zero_grad(set_to_none=True)

                indices = permutation[i: i + self.mini_batch_size]

                mini_batch_advantage = advantage[indices, :]
                if self.config['normalize_advantage']:
                    with torch.no_grad():
                        mini_batch_advantage = (mini_batch_advantage - mini_batch_advantage.mean()) / (mini_batch_advantage.std() + 1e-8)

                new_probabilities, new_values = self.training_model(self.memory.states[indices, :])
                new_values = new_values.view(new_values.shape[0], new_values.shape[1])
                if self.vf_clipping_param is not None:
                    new_values = values[indices, :] + \
                                 torch.clamp(new_values - values[indices, :],
                                             -self.vf_clipping_param, self.vf_clipping_param)
                dist = self.distribution(new_probabilities)
                new_probabilities = dist.log_prob(self.memory.actions[indices, :])
                entropy = dist.entropy().mean()

                ratio = (new_probabilities - self.memory.logprobs[indices, :]).exp()
                surr1 = ratio * mini_batch_advantage
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clipping_param, 1.0 + self.policy_clipping_param) * mini_batch_advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (computed_return[indices, :] - new_values).pow(2).mean()

                loss = actor_loss + self.config['vf_coeff'] * critic_loss - self.config['entropy_coeff'] * entropy
                loss.backward()
                if self.config['clip_grad_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), self.config['clip_grad_norm'])
                self.optimizer.step()
                with torch.no_grad():
                    mini_batch_kl = torch.mean(self.memory.logprobs[indices, :] - new_probabilities).cpu()
                    accumulated_epoch_kl_div += mini_batch_kl
                nb_mini_batches_done += 1
            number_epoch_done += 1
            accumulated_kl_div += (accumulated_epoch_kl_div / nb_mini_batches_done)
            if self.target_kl_div is not None and (accumulated_kl_div / (number_epoch_done + 1)) > 1.5 * self.target_kl_div:
                break
        wandb.log({'kl_div_iter': accumulated_kl_div / number_epoch_done, 'epoch_this_iter': number_epoch_done}, step=self.statistics.get_iteration())
        self.scheduler.step()
        self.states_tensor = self.states_tensor.to(self.inference_device, non_blocking=True)
        self.inference_model.load_state_dict(self.training_model.state_dict())
        wandb.log(self.statistics.end_update(), step=self.statistics.get_iteration_nb())