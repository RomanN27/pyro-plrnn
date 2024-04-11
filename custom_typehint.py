from typing import Protocol
import torch

class TorchDistributionMixinDist(Protocol):

    def __init__(self,*distribution_parameters: torch.Tensor):
        ...

    def sample(self, *args, **kwargs):
        ...

    def log_prob(self, *args, **kwargs):
        ...

    def mask(self, *args, **kwargs):
        ...
