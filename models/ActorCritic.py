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
    def actor_only(self, state):
        pass

    @abstractmethod
    def critic_only(self, state):
        pass

    @abstractmethod
    def get_actor(self):
        pass

    @abstractmethod
    def get_critic(self):
        pass

    @abstractmethod
    def get_common(self):
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

    def actor_only(self, state):
        x = self.common_network(state)
        return self.actor_network(x)

    def critic_only(self, state):
        x = self.common_network(state)
        return self.value_network(x)

    def get_actor(self):
        return self.actor_network

    def get_critic(self):
        return self.value_network

    def get_common(self):
        return self.common_network
