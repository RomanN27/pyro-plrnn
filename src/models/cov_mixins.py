from typing import Protocol, Callable
from abc import ABC, abstractmethod
import torch

import torch.nn as nn


class HasForwardProtocol(Protocol):

    def forward(self, z: torch.Tensor): ...


class SimpleCovHasForwardProtocol(HasForwardProtocol):

    def get_cov_extended_forward(self, old_forward) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: ...


class SimpleCovarianceMixin(ABC):

    def __init__(self: SimpleCovHasForwardProtocol, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        old_forward = self.forward
        self.forward = self.get_cov_extended_forward(old_forward)

    def get_cov_extended_forward(self, old_forward):
        def cov_extended_forward(z):  # I am not understanding why this one doesnt need a self like first argument
            return old_forward(z), self.sigma

        return cov_extended_forward


class FixedCovariance:

    def __init__(self: HasForwardProtocol, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        old_forward = self.forward
        self.forward = self.get_cov_extended_forward(old_forward)

    def get_cov_extended_forward(self, old_forward):
        def cov_extended_forward(z):  # I am not understanding why this one doesnt need a self like first argument
            return old_forward(z), self.sigma

        return cov_extended_forward
