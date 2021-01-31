import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, activation=F.relu, hidden_layers=[128, 128]):
        super(MLP, self).__init__()
        assert len(hidden_layers) > 0
        layers = [nn.Linear(in_dim, hidden_layers[0])]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.linear_layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return self.linears[-1](x)