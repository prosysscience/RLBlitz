import torch.nn as nn

from models.MLP import MLP
from abc import ABC, abstractmethod


class AbstractActorCritic(nn.Module, ABC):

    @abstractmethod
    def __init__(self, config, state_dim, action_dim):
        super().__init__()

    @abstractmethod
    def forward(self, state):
        pass

    @abstractmethod
    def actor(self, state):
        pass

    @abstractmethod
    def critic(self, state):
        pass


class ActorCritic(AbstractActorCritic):
    def __init__(self, config, state_dim, action_dim):
        super(ActorCritic, self).__init__(config, state_dim, action_dim)
        activation = config['activation_fn']
        hidden_layers = config['nn_architecture']
        logistic_function = config['logistic_function']
        # actor
        self.actor_network = MLP(state_dim, action_dim, activation, hidden_layers, logistic_function)
        # critic
        self.value_network = MLP(state_dim, 1, activation, hidden_layers)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def actor(self, state):
        return self.actor_network(state)

    def critic(self, state):
        return self.value_network(state)
