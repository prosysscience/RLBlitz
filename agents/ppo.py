import torch
import wandb

from agents.a2c import A2C
from configs.ppo_default import default_ppo_config


class PPO(A2C):

    def __init__(self, config=default_ppo_config):
        super().__init__(config)
        self.ppo_epoch = config['ppo_epochs']
        self.clipping_param = config['clipping_param']
        self.mini_batch_size = config['mini_batch_size']
        self.max_kl_div = config['max_kl_div']

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
        with torch.no_grad():
            advantage = self._compute_advantage(self.config['use_gae'])
            if self.config['normalize_advantage']:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        for epoch in range(self.ppo_epoch):
            with torch.no_grad():
                permutation = torch.randperm(self.memory.states.size(0) - 1)
            for i in range(0, self.memory.states.size(0) - 1, self.mini_batch_size):
                self.optimizer.zero_grad(set_to_none=True)

                indices = permutation[i: i + self.mini_batch_size]

                new_probabilities, new_values = self.training_model(self.memory.states[indices, :])
                new_values = new_values.view(new_values.shape[0], new_values.shape[1])
                dist = self.distribution(new_probabilities)
                new_probabilities = dist.log_prob(self.memory.actions[indices, :])
                entropy = dist.entropy().mean()

                ratio = (new_probabilities - self.memory.logprobs[indices, :].detach()).exp()
                surr1 = ratio * advantage[indices, :]
                surr2 = torch.clamp(ratio, 1.0 - self.clipping_param, 1.0 + self.clipping_param) * advantage[indices, :]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (self.memory.rewards[indices, :] - new_values).pow(2).mean()

                loss = actor_loss + self.config['vf_coeff'] * critic_loss - self.config['entropy_coeff'] * entropy
                loss.backward()
                if self.config['clip_grad_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), self.config['clip_grad_norm'])
                self.optimizer.step()
        self.scheduler.step()
        self.states_tensor = self.states_tensor.to(self.inference_device, non_blocking=True)
        self.inference_model.load_state_dict(self.training_model.state_dict())
        wandb.log(self.statistics.end_update(), step=self.statistics.get_iteration_nb())