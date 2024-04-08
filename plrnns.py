import math

import torch
from torch import nn as nn
from abc import ABC, abstractmethod
from functools import partial

class PLRNN(nn.Module):

    def __init__(self,z_dim: int, connectivity_module: nn.Module, cov_module: nn.Module):
        super().__init__()
        self.z_dim = z_dim
        self.diag = Diagonal(z_dim)
        self.bias = nn.Parameter(torch.zeros(z_dim))
        self.connectivity_module = connectivity_module
        self.cov_module = cov_module

    def forward(self,z):
        loc = self.diag(z) + self.connectivity_module(z) + self.bias
        cov = self.cov_module(z)
        return loc, cov

class LinearObservationModel(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, obs_dim, z_dim):
        super().__init__()
        # initialize the three linear.yaml transformations used in the neural network
        self.linear = nn.Linear(z_dim, obs_dim)
        self.Gamma  = nn.Parameter(torch.empty(obs_dim))
        nn.init.uniform_(self.Gamma,0,0.1**0.5)


    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        return self.linear(z_t), self.Gamma**2


class Diagonal(nn.Module):
    def __init__(self,z_dim: int):
        super().__init__()
        self.A_diag = nn.Parameter(torch.Tensor(1,z_dim))

        nn.init.normal_(self.A_diag,std=0.1)  # weight init
    def forward(self,z):
        return self.A_diag* z


class OffDiagonal(nn.Module):
    def __init__(self,z_dim:int,):
        super().__init__()
        self.W = nn.Parameter(torch.empty(z_dim,z_dim))
        self.mask = 1 - torch.eye(z_dim)
        nn.init.xavier_normal_(self.W,gain=0.1)
    def forward(self,z):
        return torch.matmul(z,self.W*self.mask)

class Bias(nn.Module):
    def __init__(self,z_dim: int):
        self.b = nn.Parameter(torch.zeros(z_dim))

    def __call__(self, z):
        return z + self.b




class ConstantCovariance(nn.Module):
    def __init__(self,z_dim):
        super().__init__()
        self.Sigma = nn.Parameter(torch.ones(z_dim))
    def forward(self,z):
        return self.Sigma.expand(len(z), *self.Sigma.shape) ** 2

class LinearCovariance(nn.Module):
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
        return self.sigmoid(cov) * self.max_

class OffDiagonalConnector(nn.Module):

    def __init__(self,z_dim,phi: nn.Module):
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




class ShallowConnector(nn.Module):
    def __init__(self,z_dim,hidden_dim):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim,z_dim, bias=False))
    def forward(self, z):
        return self.sequential(z)




class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim
        # initialize the three linear.yaml transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.ReLU()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale +0.01

if __name__ == "__main__":
    B = 12
    z_dim  =5
    dend = DendriticPhi(z_dim,B)
    batch_size = 32
    z  =torch.randn(batch_size, z_dim)
    dend(z)