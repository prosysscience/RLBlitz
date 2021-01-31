import torch.nn as nn
from torch.distributions import Categorical


class SoftmaxCategorical:

    def __init__(self, fn=nn.Softmax(dim=1), distribution=Categorical):
        self.fn = fn
        self.distribution = distribution

    def sample(self, logit):
        probabilities = self.fn(logit)
        return self.distribution(probabilities)
