import torch
from torch import nn as nn


class ConstantCovariance(nn.Module):
    def __init__(self,z_dim: int, initial_sigma: float=1.):
        super().__init__()
        self.Sigma = nn.Parameter(torch.ones(z_dim))
        self.initial_sigma = initial_sigma
    def forward(self,z: torch.Tensor):
        return self.Sigma ** 2 * self.initial_sigma

class FixedCovariance(nn.Module):

    def __init__(self,sigma: float):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma),requires_grad=False)

    def forward(self, z:torch.Tensor):
        return self.sigma

class LinearCovariance(nn.Module):
    def __init__(self,z_dim: int):
        super().__init__()
        self.Sigma = nn.Linear(z_dim, z_dim)
        #unit_matrix = torch.eye(z_dim, z_dim)
        #with torch.no_grad():  # To avoid tracking gradients for this operation
        #    self.Sigma.weight.copy_(unit_matrix)
        self.sigmoid = nn.Sigmoid()
        self.max_ = nn.Parameter(torch.ones(1))
    def forward(self,z:torch.Tensor):
        cov = self.Sigma(z)
        return self.sigmoid(cov) * self.max_ + self.max_/10e5
