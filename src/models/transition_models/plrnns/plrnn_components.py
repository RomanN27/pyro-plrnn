from abc import ABC, abstractmethod

import torch
from torch import nn as nn


class Diagonal(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.A_diag = nn.Parameter(torch.empty(z_dim))

    def forward(self, z):
        return self.A_diag * z


class OffDiagonal(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(z_dim, z_dim))
        self.register_buffer("mask", 1 - torch.eye(z_dim))

    def forward(self, z):
        return torch.matmul(z, self.W * self.mask)





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



class ShallowPhi(nn.Module):
    def __init__(self,z_dim,hidden_dim):
        super().__init__()
        self.linear_1 = nn.Linear(z_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.z_dim = z_dim

    def forward(self, z):
        return self.relu(self.linear_1(z))[...,:self.z_dim]

class ClippedShallowPhi(ShallowPhi):

    def forward(self,z):
        #TODO Write doc here
        x = self.linear_1(z)
        x_2 =  x - self.linear_1.bias
        x = self.relu(x)
        x = x - self.relu(x_2)
        return x[...,:self.z_dim]






