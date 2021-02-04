import time
import numpy as np


class Statistics:

    def __init__(self, num_worker):
        self.num_worker = num_worker
        self.episode_return = np.zeros((num_worker, ), dtype=np.float)
        self.episode_len = np.zeros((num_worker, ), dtype=np.float)
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

    def add_rewards(self, rewards):
        self.episode_return += rewards
        self.episode_len += 1

    def episode_done(self, worker_id):
        self.episode_return[worker_id] = 0
        self.episode_len[worker_id] = 0
        self.episode_number += 1
        self.episode_this_iter += 1

    '''
    Finer control here
    '''
    def start_step(self):
        self.time_start_step = time.time()

    def end_step(self):
        self.total_step += self.num_worker

    def start_act(self):
        self.time_start_act = time.time()

    def end_act(self):
        pass

    def start_update(self):
        self.time_start_update = time.time()

    def end_update(self):
        pass

    def start_train(self):
        self.time_start_train = time.time()

    def end_train(self):
        self.iteration += 1
