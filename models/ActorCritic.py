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

        self.common_layers = config['common_layers'](state_dim)

        if self.common_layers is not None:
            self.actor_network = nn.Sequential(
                self.common_layers,
                config['actor_layers'](state_dim, action_dim)
            )
            self.value_network = nn.Sequential(
                self.common_layers,
                config['critic_layers'](state_dim)
            )
        else:
            self.actor_network = config['actor_layers'](state_dim, action_dim)
            self.value_network = config['critic_layers'](state_dim)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def actor(self, state):
        return self.actor_network(state)

    def critic(self, state):
        return self.value_network(state)
