import torch

from abc import ABC, abstractmethod

class ParameterScheduler(ABC):

    def __init__(self, start_val, end_val, start_step, end_step):
        self.start_val = start_val
        self.end_val = end_val
        self.start_step = start_step
        self.end_step = end_step
        self.step = 0

    def inc_step(self, nb_step=1):
        self.step += nb_step

    @abstractmethod
    def get_current_value(self):
        pass

    def __float__(self):
        return self.get_current_value()

    def __double__(self):
        return self.get_current_value()

    def __int__(self):
        return self.get_current_value()

    def __long__(self):
        return self.get_current_value()

class ConstantScheduler(ParameterScheduler):

    def __init__(self, start_val, end_val=None, start_step=None, end_step=None):
        super().__init__(start_val, end_val, start_step, end_step)

    def get_current_value(self):
        return self.start_val

class LinearDecayScheduler(ParameterScheduler):

    def __init__(self, start_val, end_val, start_step, end_step):
        super().__init__(start_val, end_val, start_step, end_step)
        self.slope = (end_val - start_val) / (end_step - start_step)

    def get_current_value(self):
        if self.step < self.start_step:
            return self.start_val
        elif self.step > self.end_step:
            return self.end_val
        return max(self.slope * (self.step - self.start_step) + self.start_val, self.end_val)

class RateDecayScheduler(ParameterScheduler):

    def __init__(self, start_val, end_val, start_step, end_step, decay_rate=0.9, frequency=20.):
        super().__init__(start_val, end_val, start_step, end_step)
        self.decay_rate = decay_rate
        self.frequency = frequency

    def get_current_value(self):
        if self.step < self.start_step:
            return self.start_val
        elif self.step >= self.end_step:
            return self.end_val
        step_per_decay = (self.end_step - self.start_step) / self.frequency
        decay_step = (self.step - self.start_step) / step_per_decay
        return max(torch.pow(self.decay_rate, decay_step) * self.start_val, self.end_val)

class PeriodicDecayScheduler(ParameterScheduler):

    def __init__(self, start_val, end_val, start_step, end_step, frequency=60.):
        super().__init__(start_val, end_val, start_step, end_step)
        self.frequency = frequency

    def get_current_value(self):
        if self.step < self.start_step:
            return self.start_val
        elif self.step >= self.end_step:
            return self.end_val
        step_per_decay = (self.end_step - self.start_step) / self.frequency
        x = (self.step - self.start_step) / step_per_decay
        unit = self.start_val - self.end_val
        val = self.end_val * 0.5 * unit * (1 + torch.cos(x) * (1 - x / self.frequency))
        return max(val, self.end_val)
