import time
import numpy as np


class Statistics:

    def __init__(self, config):
        self.config = config
        self.num_worker = self.config['num_worker']
        self.episode_return = np.zeros((self.num_worker, ), dtype=np.float)
        self.episode_len = np.zeros((self.num_worker, ), dtype=np.float)
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

    def get_iteration_nb(self):
        return self.iteration

    def add_rewards(self, rewards):
        self.episode_return += rewards
        self.episode_len += 1
        return {}

    def episode_done(self, worker_id):
        result = {'episode_return': self.episode_return[worker_id],
                'episode_len': self.episode_len[worker_id],
                'episode_number': self.episode_number}
        self.episode_return[worker_id] = 0
        self.episode_len[worker_id] = 0
        self.episode_number += 1
        self.episode_this_iter += 1
        return result

    '''
    Finer control here
    '''
    def start_step(self):
        self.time_start_step = time.time()
        return {}

    def end_step(self):
        self.total_step += self.num_worker
        return {'step_time': time.time() - self.time_start_step}

    def start_act(self):
        self.time_start_act = time.time()
        return {}

    def end_act(self):
        return {'acting_time': time.time() - self.time_start_act}

    def start_update(self):
        self.time_start_update = time.time()
        return {}

    def end_update(self):
        return {'training_time': time.time() - self.time_start_update}

    def start_train(self):
        self.time_start_train = time.time()
        self.episode_this_iter = 0
        return {}

    def end_train(self):
        self.iteration += 1
        return {'iteration': self.iteration,
                   'episodes_this_iter': self.episode_this_iter,
                   'total_steps': self.total_step,
                   'iteration_time': time.time() - self.time_start_train}
