from collections import deque

import numpy as np


class RunningMetrics:

    def __init__(self, config):
        self.config = config
        self.metric_smoothing = config['metric_smoothing']
        self.data = deque(maxlen=self.metric_smoothing)
        self.min = float('inf')
        self.max = float('-inf')
        self.sum = 0
        self.len = 0

    def add(self, value):
        if self.len < self.metric_smoothing:
            self.data.append(value)
            self.len += 1
            self.sum += value
            self.min = min(self.min, value)
            self.max = max(self.max, value)
        else:
            old_value = self.data.popleft()
            self.data.append(value)
            self.sum += (value - old_value)
            if self.min == old_value:
                self.min = np.min(self.data)
            if self.max == old_value:
                self.max = np.max(self.data)

    def get_metric(self):
        return self.sum / self.len, self.min, self.max
