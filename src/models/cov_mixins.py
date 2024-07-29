from typing import Protocol, Callable
import torch
import torch.nn as nn


class HasForwardProtocol(Protocol):
    def forward(self, z: torch.Tensor): ...


class BaseCovarianceMixin(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extend_forward(self, old_forward: Callable, covariance: Callable) -> Callable:
        def cov_extended_forward(z):
            return old_forward(z), covariance(z)

        return cov_extended_forward


class FixedCovarianceMixin(BaseCovarianceMixin):

    def __init__(self, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        old_forward = self.forward
        self.forward = self.extend_forward(old_forward, lambda z: self.sigma)


class ConstantCovarianceMixin(BaseCovarianceMixin):
    def __init__(self, sigma: float, z_dim: int, **kwargs):
        super().__init__(z_dim=z_dim, **kwargs)

        self.Sigma = nn.Parameter(torch.ones(z_dim))
        self.initial_sigma = sigma
        old_forward = self.forward
        self.forward = self.extend_forward(old_forward, lambda z: self.Sigma ** 2 * self.initial_sigma)

class LinearCovarianceMixin(BaseCovarianceMixin):
    def __init__(self,z_dim: int, **kwargs):
        super().__init__(z_dim = z_dim,**kwargs)
        self.sigma_layer = nn.Linear(z_dim, z_dim)
        self.sigmoid = nn.Sigmoid()
        self.max_ = nn.Parameter(torch.ones(1))
        old_forward = self.forward
        self.forward = self.extend_forward(old_forward,self.forward)


    def forward(self,z:torch.Tensor):
        cov = self.sigma_layer(z)
        return self.sigmoid(cov) * self.max_ + self.max_/10e5

class SimpleModule(nn.Module):
    def __init__(self, a, b, z_dim):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.z_dim = z_dim

    def forward(self, z):
        print(self.z_dim)
        return z * self.b + self.a



class dendPLRNNInitialization(Initializer):
    """
    Implementation of initialiazion proposed in :
    Brenner, M., Hess, F., Mikhaeil, J. M., Bereska, L., Monfared, Z., Kuo, P.-C., & Durstewitz, D. (2022).
    _Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical Systems_ (arXiv:2207.02542). arXiv.
    [https://doi.org/10.48550/arXiv.2207.02542](https://doi.org/10.48550/arXiv.2207.02542)"""

