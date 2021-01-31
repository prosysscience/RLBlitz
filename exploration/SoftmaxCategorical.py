import torch.nn as nn
from torch.distributions import Categorical


class SoftmaxCategorical:

    def __init__(self, fn=nn.Softmax(dim=1), distribution=Categorical):
        self.fn = fn
        self.distribution = distribution

    def sample(self, logits):
        probabilities = self.fn(logits)
        return self.distribution(probabilities)

    def __str__(self):
        return 'Function applied to logit {}, and distribution {}'.format(self.fn, self.distribution)
