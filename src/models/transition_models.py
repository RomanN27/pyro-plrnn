from typing import TYPE_CHECKING

import torch
from torch import nn as nn
import pyro
from src.models.elementary_components import Diagonal, OffDiagonal
from lightning import LightningModule

from src.models.model_sampler import ModelBasedSampler

if TYPE_CHECKING:
    from pyro.distributions import Distribution


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

    z_dims = [1,2,3,9]
    dends = [DendriticConnector(z_dim,10) for z_dim in z_dims]
    covs = [ConstantCovariance(z_dim) for z_dim in z_dims]
    plrnnns = [PLRNN(z_dim,dend,cov) for z_dim,dend,cov in zip(z_dims,dends,covs)]
    transition_samplers = [ModelBasedSampler(plrnn, pyro.distributions.Normal) for plrnn in plrnnns]
    prod = SkewProduct(*transition_samplers[:-1], target_model=transition_samplers[-1])

    z_0  = torch.normal(1,1,size=(1,z_dims[-1]))
    z = [z_0]
    dt = 0.1
    for _ in range(1000):
        new_z = prod(z[-1]) *dt + z[-1]
        z.append(new_z)


    Z =torch.cat(z)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    trajectory = Z.detach().numpy()

    # Step 3: Plot the 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = trajectory[:, -3]
    y = trajectory[:, -2]
    z = trajectory[:, -1]

    # Create a colormap based on time
    colors = cm.viridis(np.linspace(0, 1, len(x)))

    # Plot each segment with a different color
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=colors[i])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

    plt.plot(trajectory[:, 0])
    plt.plot(trajectory[:, 1])
    plt.plot(trajectory[:, 2])

    plt.plot(trajectory[:, -3])
    plt.plot(trajectory[:,-2])
    plt.plot(trajectory[:, -1])