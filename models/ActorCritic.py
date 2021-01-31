import torch.nn as nn
import torch.nn.functional as F

from exploration.SoftmaxCategorical import SoftmaxCategorical
from models.MLP import MLP


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu, hidden_layers=[128, 128],
                 exploration=SoftmaxCategorical):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = MLP(state_dim, action_dim, activation, hidden_layers)

        # critic
        self.value_layer = MLP(state_dim, 1, activation, hidden_layers)

        self.exploration = exploration

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def actor(self, state):
        prob = self.actor(state)
        return self.exploration(prob)

    def critic(self, state):
        return self.critic(state)
