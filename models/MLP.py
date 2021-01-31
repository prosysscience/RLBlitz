import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, activation=nn.ReLU(), hidden_layers=[128, 128]):
        super(MLP, self).__init__()
        assert len(hidden_layers) > 0
        model_layers = [nn.Linear(in_dim, hidden_layers[0]), activation]
        for i in range(len(hidden_layers) - 1):
            model_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            model_layers.append(activation)
        model_layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*model_layers)
        self.activation = activation

    def forward(self, x):
        return self.model(x)