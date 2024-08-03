from typing import Protocol, Callable
import torch
import torch.nn as nn


class BaseCovarianceMixin:
    def __init__(self:nn.Module, **kwargs):
        super().__init__(**kwargs)

    def extend_forward(self:nn.Module, old_forward: Callable, covariance: Callable) -> Callable:
        def cov_extended_forward(z):
            return old_forward(z), covariance(z)

        return cov_extended_forward


class FixedCovarianceMixin(BaseCovarianceMixin):

    def __init__(self:nn.Module, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        old_forward = self.forward
        self.forward = self.extend_forward(old_forward, lambda z: self.sigma)


class ConstantCovarianceMixin(BaseCovarianceMixin):
    def __init__(self:nn.Module, sigma: float, z_dim: int, **kwargs):
        super().__init__(z_dim=z_dim, **kwargs)

        self.sigma = nn.Parameter(torch.tensor(sigma))
        old_forward = self.forward

        self.forward = self.extend_forward(old_forward, lambda z:  torch.abs(self.sigma) + 1e-5)

class LinearCovarianceMixin(BaseCovarianceMixin):
    def __init__(self:nn.Module,z_dim: int, **kwargs):
        super().__init__(z_dim = z_dim,**kwargs)
        self.sigma_layer = nn.Linear(z_dim, z_dim)
        self.sigmoid = nn.Sigmoid()
        self.max_ = nn.Parameter(torch.ones(1))
        old_forward = self.forward
        self.forward = self.extend_forward(old_forward,self.forward)


    def forward(self,z:torch.Tensor):
        cov = self.sigma_layer(z)
        return self.sigmoid(cov) * self.max_ + self.max_/10e5


