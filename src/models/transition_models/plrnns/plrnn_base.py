import torch
from torch import nn as nn

from src.models.elementary_components import Diagonal


class PLRNN(nn.Module):

    def __init__(self,z_dim: int, connectivity_module: nn.Module):
        super().__init__()
        self.z_dim = z_dim
        self.diag = Diagonal(z_dim)
        self.bias = nn.Parameter(torch.zeros(z_dim))
        self.connectivity_module = connectivity_module


    def forward(self,z):
        z_new = self.diag(z)
        z_new += self.connectivity_module(z)
        z_new += self.bias

        return z_new
