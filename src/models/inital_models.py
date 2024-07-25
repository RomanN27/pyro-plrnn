from lightning import LightningModule
import torch.nn as nn

import torch
class SimpleInit(LightningModule):

    def __init__(self,z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.cov = nn.Parameter(torch.ones(z_dim))
        self.mu = nn.Parameter(torch.zeros(z_dim))


    def forward(self,z):

        return self.mu, self.cov
