from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def render(self, number_worker):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_agent_checkpoint(self, filename='default_checkpoint.pickle'):
        pass

    @classmethod
    def load_agent_checkpoint(cls, filename='default_checkpoint.pickle'):
        pass
