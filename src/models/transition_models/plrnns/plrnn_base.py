import torch
from torch import nn as nn
from typing import Callable
from src.models.transition_models.plrnns.plrnn_components import Diagonal, OffDiagonal
from abc import ABC, abstractmethod
from typing import Generic,TypeVar

T = TypeVar("T", bound=Callable[[torch.Tensor], torch.Tensor])

class PLRNN(Generic[T],nn.Module):

    def __init__(self, z_dim: int, phi: T):
        super().__init__()
        self.z_dim = z_dim
        self.diag = Diagonal(z_dim)
        self.off_diag = OffDiagonal(z_dim)
        self.bias = nn.Parameter(torch.empty(z_dim))
        self.phi= phi


    def forward(self,z):

        return  self.diag(z) +  self.off_diag(self.phi(z)) + self.bias
