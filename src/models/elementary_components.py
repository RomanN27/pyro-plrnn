import torch
from torch import nn as nn
from lightning import LightningModule

class Diagonal(LightningModule):
    def __init__(self,z_dim: int):
        super().__init__()
        self.A_diag = nn.Parameter(torch.ones(z_dim) )

        #nn.init.normal_(self.A_diag,std=0.1)  # weight init
    def forward(self,z):
        return self.A_diag* z


class OffDiagonal(LightningModule):
    def __init__(self,z_dim:int):
        super().__init__()
        self.W = nn.Parameter(torch.ones(z_dim,z_dim) )
        self.register_buffer("mask", 1 - torch.eye(z_dim))


    def forward(self,z):
        return torch.matmul(z,self.W*self.mask)


class Bias(LightningModule):
    def __init__(self,z_dim: int):
        self.b = nn.Parameter(torch.zeros(z_dim))

    def __call__(self, z):
        return z + self.b

