import torch
from torch import nn as nn

from src.models.elementary_components import OffDiagonal


class OffDiagonalConnector(nn.Module):

    def __init__(self,z_dim: int,phi: nn.Module):
        super().__init__()
        self.off_diag = OffDiagonal(z_dim)
        self.phi = phi

    def forward(self,z):
        return  self.off_diag(self.phi(z))


class DendriticPhi(nn.Module):
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
        return (self.relu(z.unsqueeze(-1) - self.H) - self.relu(z.unsqueeze(-1))) @ self.alpha


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


class ShallowConnector(nn.Module):
    def __init__(self,z_dim,hidden_dim):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim,z_dim, bias=False))
    def forward(self, z):
        return self.sequential(z)
