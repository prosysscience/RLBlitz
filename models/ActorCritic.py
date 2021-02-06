import torch.nn as nn

from abc import ABC, abstractmethod


class AbstractActorCritic(nn.Module, ABC):

    @abstractmethod
    def __init__(self, state_dim, action_dim, **kwargs):
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
    def __init__(self, state_dim, action_dim, common_layers, actor_layers, critic_layers):
        super(ActorCritic, self).__init__(state_dim, action_dim)

        self.common_network = common_layers(state_dim)
        self.actor_network = actor_layers(state_dim, action_dim)
        self.value_network = critic_layers(state_dim)

    def forward(self, state):
        x = self.common_network(state)
        return self.actor_network(x), self.value_network(x)

    def actor(self, state):
        x = self.common_network(state)
        return self.actor_network(x)

    def critic(self, state):
        x = self.common_network(state)
        return self.value_network(x)
