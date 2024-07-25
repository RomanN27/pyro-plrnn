import torch
from torch import nn as nn
from lightning import LightningModule

class Diagonal(LightningModule):
    def __init__(self,z_dim: int):
        super().__init__()
        self.A_diag = nn.Parameter(torch.Tensor(1,z_dim))

        nn.init.normal_(self.A_diag,std=0.1)  # weight init
    def forward(self,z):
        return self.A_diag* z


class OffDiagonal(LightningModule):
    def __init__(self,z_dim:int,):
        super().__init__()
        self.W = nn.Parameter(torch.empty(z_dim,z_dim))
        self.register_buffer("mask", 1 - torch.eye(z_dim))

        nn.init.xavier_normal_(self.W,gain=0.1)
    def forward(self,z):
        return torch.matmul(z,self.W*self.mask)


class Bias(LightningModule):
    def __init__(self,z_dim: int):
        self.b = nn.Parameter(torch.zeros(z_dim))

    def __call__(self, z):
        return z + self.b

class ScaleModule(LightningModule):
    def __init__(self,output_dim: int):
        self.output_dim = output_dim
        nn.Linear(hidden_dim, n_time_steps * output_dim ** 2)
