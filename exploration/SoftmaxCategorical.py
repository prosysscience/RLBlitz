import torch.nn.functional as F
from torch.distributions import Categorical


class SoftmaxCategorical:

    def __init__(self, fn=F.softmax, distribution=Categorical):
        self.fn = fn
        self.distribution = distribution

    def dist(self, logits):
        probabilities = self.fn(logits, dim=1)
        return self.distribution(probabilities)

    def __str__(self):
        return 'Function applied to logit {}, and distribution {}'.format(self.fn, self.distribution)
