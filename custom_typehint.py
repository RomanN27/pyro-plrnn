from pyro.distributions import Distribution, TorchDistributionMixin


class TorchDistributionMixinDist(Distribution, TorchDistributionMixin):
    pass
