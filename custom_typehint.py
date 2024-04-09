from typing import Protocol


class TorchDistributionMixinDist(Protocol):
    def sample(self, *args, **kwargs):
        ...

    def log_prob(self, *args, **kwargs):
        ...

    def mask(self, *args, **kwargs):
        ...
