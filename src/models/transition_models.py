from typing import Any, Protocol, TYPE_CHECKING,Callable, Type

import torch
from torch import nn as nn
import pyro
from src.models.elementary_components import Diagonal, OffDiagonal
from lightning import LightningModule
from uuid import uuid4
if TYPE_CHECKING:
    from pyro.distributions import Distribution


random_name = lambda : str(uuid4())

class TransitionModel(Protocol):
    z_dim: int

    def __call__(self, *args, **kwargs)->tuple:...

class ModelBasedSampler(LightningModule):
    def __init__(self, transition_model: nn.Module, distribution:Type["Distribution"]):
        super().__init__()
        self.model = transition_model
        self.distribution =  distribution

    def forward(self, z: torch.Tensor, name:str = random_name()) -> torch.Tensor:

        dist_parameters = self.model(z)
        dist = self.distribution(*dist_parameters) #typecheck error because Distribuiton itself does not implement init however all subclasses do
        return pyro.sample(name,dist)




class PLRNN(LightningModule):

    def __init__(self,z_dim: int, connectivity_module: nn.Module, cov_module: nn.Module):
        super().__init__()
        self.z_dim = z_dim
        self.diag = Diagonal(z_dim)
        self.bias = nn.Parameter(torch.zeros(z_dim))
        self.connectivity_module = connectivity_module
        self.cov_module = cov_module

    def forward(self,z):
        loc = self.diag(z)
        loc += self.connectivity_module(z)
        loc += self.bias
        cov = self.cov_module(z)
        return loc, cov


class ConstantCovariance(LightningModule):
    def __init__(self,z_dim):
        super().__init__()
        self.Sigma = nn.Parameter(torch.ones(z_dim))
    def forward(self,z):
        return self.Sigma ** 2

class LinearCovariance(LightningModule):
    def __init__(self,z_dim):
        super().__init__()
        self.Sigma = nn.Linear(z_dim, z_dim)
        #unit_matrix = torch.eye(z_dim, z_dim)
        #with torch.no_grad():  # To avoid tracking gradients for this operation
        #    self.Sigma.weight.copy_(unit_matrix)
        self.sigmoid = nn.Sigmoid()
        self.max_ = nn.Parameter(torch.ones(1))
    def forward(self,z):
        cov = self.Sigma(z)
        return self.sigmoid(cov) * self.max_ + self.max_/10e5

class OffDiagonalConnector(LightningModule):

    def __init__(self,z_dim,phi: nn.Module):
        super().__init__()
        self.off_diag = OffDiagonal(z_dim)
        self.phi = phi

    def forward(self,z):
        return  self.off_diag(self.phi(z))

class DendriticPhi(LightningModule):
    def __init__(self, z_dim: int, B: int):
        super().__init__()
        self.H  = nn.Parameter(torch.zeros(1,z_dim,B))
        self.alpha = nn.Parameter(torch.empty(B))
        self.relu = nn.ReLU()
        nn.init.normal_(self.alpha)

    def forward(self,z):
        return self.relu(z.unsqueeze(-1) -self.H) @ self.alpha

class ClippedDendriticPhi(DendriticPhi):
    def forward(self,z):
        return (self.relu(z.unsqueeze(-1) - self.H) - self.relu(z)) @ self.alpha

class VanillaConnector(OffDiagonalConnector):
    def __init__(self,z_dim):
        super().__init__(z_dim, nn.ReLU())

class DendriticConnector(OffDiagonalConnector):

    def __init__(self,z_dim,B):
        dendritic_phi = DendriticPhi(z_dim,B)
        super().__init__(z_dim,phi=dendritic_phi)


class ClippedDendriticConnector(OffDiagonalConnector):

    def __init__(self, z_dim, B):
        clipped_dendritic_phi = ClippedDendriticPhi(z_dim, B)
        super().__init__(z_dim,phi=clipped_dendritic_phi)




class ShallowConnector(LightningModule):
    def __init__(self,z_dim,hidden_dim):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim,z_dim, bias=False))
    def forward(self, z):
        return self.sequential(z)


if __name__ == "__main__":
    B = 12
    z_dim  =5
    dend = DendriticPhi(z_dim,B)
    batch_size = 32
    z  =torch.randn(batch_size, z_dim)
    dend(z)