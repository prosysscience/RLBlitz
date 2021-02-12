import time
import numpy as np


class Statistics:

    def __init__(self, config):
        self.config = config
        self.num_worker = self.config['num_worker']
        self.episode_return = np.zeros((self.num_worker,), dtype=np.float)
        self.episode_len = np.zeros((self.num_worker,), dtype=np.float)
        self.episode_number = 0
        self.total_step = 0
        self.iteration = 0
        self.episode_this_iter = 0
        # Timers
        self.time_start_agent = time.time()
        self.time_start_step = 0
        self.time_start_act = 0
        self.time_start_update = 0
        self.time_start_train = 0
        self.time_start_inference = 0
        self.time_env_wait = 0

    def get_iteration_nb(self):
        return self.iteration

    def add_rewards(self, rewards):
        self.episode_return += rewards
        self.episode_len += 1
        return {}

    def episode_done(self, worker_id):
        result = {'Environment/episode_return': self.episode_return[worker_id],
                  'Environment/episode_len': self.episode_len[worker_id],
                  'Environment/episode_number': self.episode_number}
        self.episode_return[worker_id] = 0
        self.episode_len[worker_id] = 0
        self.episode_number += 1
        self.episode_this_iter += 1
        return result

    def get_iteration(self):
        return self.iteration

    '''
    Finer control here
    '''

    def start_step(self):
        self.time_start_step = time.time()
        return {}

    def end_step(self):
        self.total_step += self.num_worker
        return {'Timers/step_time': time.time() - self.time_start_step}

    def start_act(self):
        self.time_start_act = time.time()
        return {}

    def end_act(self):
        return {'Timers/acting_time': time.time() - self.time_start_act}

    def start_update(self):
        self.time_start_update = time.time()
        return {}

    def end_update(self):
        return {'Timers/training_time': time.time() - self.time_start_update}

    def start_train(self):
        self.time_start_train = time.time()
        return {}

    def end_train(self):
        episodes_done = self.episode_this_iter
        self.iteration += 1
        self.episode_this_iter = 0
        return {'Algorithm/iteration': self.iteration,
                'Algorithm/episodes_this_iter': episodes_done,
                'Environment/total_steps': self.total_step,
                'Timers/iteration_time': time.time() - self.time_start_train}

    def start_inference(self):
        self.time_start_inference = time.time()
        return {}

    def end_inference(self):
        return {'Timers/inference_time': time.time() - self.time_start_inference}

    def start_env_wait(self):
        self.time_env_wait = time.time()
        return {}

    def end_env_wait(self):
        return {'Timers/env_wait_time': time.time() - self.time_env_wait}
