import time
import numpy as np

from utils.RunningMetrics import RunningMetrics


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
        result = {'episode_return': self.episode_return[worker_id],
                  'episode_len': self.episode_len[worker_id],
                  'episode_number': self.episode_number}
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
        return {}

    def end_train(self):
        episodes_done = self.episode_this_iter
        self.iteration += 1
        self.episode_this_iter = 0
        return {'iteration': self.iteration,
                'episodes_this_iter': episodes_done,
                'total_steps': self.total_step,
                'iteration_time': time.time() - self.time_start_train}

    def start_inference(self):
        self.time_start_inference = time.time()
        return {}

    def end_inference(self):
        return {'inference_time': time.time() - self.time_start_inference}

    def start_env_wait(self):
        self.time_env_wait = time.time()
        return {}

    def end_env_wait(self):
        return {'env_wait_time': time.time() - self.time_env_wait}


class SmoothedStatistics(Statistics):

    def __init__(self, config):
        super(SmoothedStatistics, self).__init__(config)
        # Smoothing
        self.smoothing_episode_return = RunningMetrics(config)
        self.smoothing_episode_len = RunningMetrics(config)
        self.smoothing_episodes_this_iter = RunningMetrics(config)
        self.smoothing_start_step = RunningMetrics(config)
        self.smoothing_start_act = RunningMetrics(config)
        self.smoothing_start_update = RunningMetrics(config)
        self.smoothing_start_train = RunningMetrics(config)
        self.smoothing_start_inference = RunningMetrics(config)
        self.smoothing_env_wait = RunningMetrics(config)

    def episode_done(self, worker_id):
        self.smoothing_episode_return.add(self.episode_return[worker_id])
        self.smoothing_episode_len.add(self.episode_len[worker_id])
        mean_episode_return, min_episode_return, max_episode_return = self.smoothing_episode_return.get_metric()
        mean_episode_len, min_episode_len, max_episode_len = self.smoothing_episode_len.get_metric()
        result = {'episode_return': self.episode_return[worker_id],
                  'mean_episode_return': mean_episode_return,
                  'min_episode_return': min_episode_return,
                  'max_episode_return': max_episode_return,
                  'episode_len': self.episode_len[worker_id],
                  'mean_episode_len': mean_episode_len,
                  'min_episode_len': min_episode_len,
                  'max_episode_len': max_episode_len,
                  'episode_number': self.episode_number}
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
        time_elapsed = time.time() - self.time_start_step
        self.smoothing_start_step.add(time_elapsed)
        mean_start_step, min_start_step, max_start_step = self.smoothing_start_step.get_metric()
        self.total_step += self.num_worker
        return {'step_time': time_elapsed,
                'mean_start_step': mean_start_step,
                'min_start_step': min_start_step,
                'max_start_step': max_start_step}

    def start_act(self):
        self.time_start_act = time.time()
        return {}

    def end_act(self):
        time_elapsed = time.time() - self.time_start_act
        self.smoothing_start_act.add(time_elapsed)
        mean_start_act, min_start_act, max_start_act = self.smoothing_start_act.get_metric()
        return {'acting_time': time_elapsed,
                'mean_acting_time': mean_start_act,
                'min_acting_time': min_start_act,
                'max_acting_time': max_start_act}

    def start_update(self):
        self.time_start_update = time.time()
        return {}

    def end_update(self):
        time_elapsed = time.time() - self.time_start_update
        self.smoothing_start_update.add(time_elapsed)
        mean_start_update, min_start_update, max_start_update = self.smoothing_start_inference.get_metric()
        return {'training_time': time_elapsed,
                'mean_training_time': mean_start_update,
                'min_training_time': min_start_update,
                'max_training_time': max_start_update}

    def start_train(self):
        self.time_start_train = time.time()
        return {}

    def end_train(self):
        episodes_done = self.episode_this_iter
        self.smoothing_episodes_this_iter.add(episodes_done)
        mean_episodes_this_iter, min_episodes_this_iter, max_episodes_this_iter = self.smoothing_episodes_this_iter.get_metric()
        time_elapsed = time.time() - self.time_start_train
        self.smoothing_start_train.add(time_elapsed)
        mean_start_train, min_start_train, max_start_train = self.smoothing_start_train.get_metric()
        self.iteration += 1
        self.episode_this_iter = 0
        return {'episodes_this_iter': episodes_done,
                'mean_episodes_this_iter': mean_episodes_this_iter,
                'min_episodes_this_iter': min_episodes_this_iter,
                'max_episodes_this_iter': max_episodes_this_iter,
                'iteration_time': time_elapsed,
                'mean_iteration_time': mean_start_train,
                'min_iteration_time': min_start_train,
                'max_iteration_time': max_start_train,
                'total_steps': self.total_step}

    def start_inference(self):
        self.time_start_inference = time.time()
        return {}

    def end_inference(self):
        time_elapsed = time.time() - self.time_start_inference
        self.smoothing_start_inference.add(time_elapsed)
        mean_start_inference, min_start_inference, max_start_inference = self.smoothing_start_inference.get_metric()
        return {'inference_time': time_elapsed,
                'mean_inference_time': mean_start_inference,
                'min_inference_time': min_start_inference,
                'max_inference_time': max_start_inference}

    def start_env_wait(self):
        self.time_env_wait = time.time()
        return {}

    def end_env_wait(self):
        time_elapsed = time.time() - self.time_env_wait
        self.smoothing_env_wait.add(time_elapsed)
        mean_env_wait, min_env_wait, max_env_wait = self.smoothing_env_wait.get_metric()
        return {'env_wait': time_elapsed,
                'mean_env_wait': mean_env_wait,
                'min_env_wait': min_env_wait,
                'max_env_wait': max_env_wait}
