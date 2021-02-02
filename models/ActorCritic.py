import torch.nn as nn

from models.MLP import MLP


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, activation=nn.ReLU(), hidden_layers=[64, 64],
                 logistic_function=nn.Softmax(dim=1)):
        super(ActorCritic, self).__init__()
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
