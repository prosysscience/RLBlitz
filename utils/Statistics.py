import numpy as np

class Statistics:

    def __init__(self, num_worker):
        self.episode_return = np.zeros((num_worker, ), dtype=np.float)
        self.episode_number = 0
        self.total_step = 0
        self.iteration = 0