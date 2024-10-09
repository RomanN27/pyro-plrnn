import pyro
import torch
import pyro.distributions as dist
from src.utils.variable_time_series_length_utils import pad_and_reverse, collate_fn
from torch import nn as nn
from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from pyro.distributions import TorchDistributionMixin
from abc import abstractmethod
from torch.utils.data import Dataset
from lightning import LightningModule
from src.utils.custom_typehint import TensorIterable
from src.models.model_sampler import ModelBasedSampler
from pyro.distributions import MultivariateNormal, Delta, Normal
from src.utils.variable_group_enum import V
class Guide:
    @property
    def data_set(self) -> Dataset:
        raise NotImplementedError()

    @abstractmethod
    def ___call__(self, batch):
        pass

    def __call__(self, batch_indices: list[int]):
        batch = [self.data_set[ind] for ind in batch_indices]
        return self.___call__(batch)

class Combiner(LightningModule):
    """
    Parameterizes `q(z_t | z_{delta_t-1}, x_{delta_t:T})`, which is the basic building block
    of the variational_distribution (i.e. the variational distribution). The dependence on `x_{delta_t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.ReLU()

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale + 0.01

class InitialStateGuide(LightningModule):
    """
    Guide to parameterize q(z_q_0 | x_{0:T}) as a mixture of Gaussians
    """
    def __init__(self, input_dim, z_dim, num_components):
        super().__init__()
        self.z_dim = z_dim
        self.num_components = num_components
        self.lin_input_to_loc = nn.Linear(input_dim, z_dim * num_components)
        self.lin_input_to_scale = nn.Linear(input_dim, z_dim * num_components)
        self.lin_input_to_weights = nn.Linear(input_dim, num_components)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        locs = self.lin_input_to_loc(x).view(-1, self.num_components, self.z_dim)
        scales = self.softplus(self.lin_input_to_scale(x)).view(-1, self.num_components, self.z_dim)
        logits = self.lin_input_to_weights(x)
        mixing_weights = self.softmax(logits)
        return locs, scales, mixing_weights

class RNNGuide(LightningModule):
    def __init__(self, rnn: nn.RNN, combiner: Combiner, dist: Type["TorchDistributionMixin"]):
        super().__init__()
        self.rnn = rnn
        self.combiner = combiner
        self.dist = dist
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn.hidden_size))

    def forward(self, batch: torch.Tensor):
        reversed_batch = batch.flip(1)
        pyro.module("dmm", self)

        rnn_output, _ = self.rnn(reversed_batch)


class SimpleNormalNNGuide(LightningModule):

    def __init__(self, n_time_steps: int, input_dim: int, output_dim: int , hidden_dim):
        super().__init__()
        self.output_dim= output_dim
        self.lin_1 = nn.Linear(n_time_steps*input_dim,hidden_dim)
        self.relu_1 = nn.ReLU()
        self.scale_layer = nn.Linear(hidden_dim, n_time_steps * output_dim ** 2)
        self.softplus= nn.Softplus()
        self.mu_layer = nn.Linear(hidden_dim, n_time_steps * output_dim)


    def forward(self, batch: torch.Tensor):
        x = self.relu_1(self.lin_1(batch.reshape(batch.size(0),-1 )))
        scale = self.scale_layer(x).reshape(batch.size(0),batch.size(1),self.output_dim,self.output_dim)
        scale_diag = torch.diagonal(scale,dim1=-2,dim2=-1)
        scale_diag = self.softplus(scale_diag)
        scale_diag = torch.diag_embed(scale_diag)
        scale_lower_tria = torch.tril(scale,diagonal=-1)
        scale = scale_lower_tria + scale_diag

        mu = self.mu_layer(x).reshape(batch.size(0),batch.size(1),self.output_dim)
        Z =[]
        for t, (mu_t, scale_t) in enumerate(zip(torch.split(mu,1,1), torch.split(scale,1,1))):
            normal = MultivariateNormal(mu_t.squeeze(1), scale_tril=scale_t.squeeze(1))
            z = pyro.sample(f"z_{t+1}",normal.to_event(1))
            Z.append(z)
        return Z


class IdentityGuide(LightningModule):
    def __init__(self,z_dim:int):
        super().__init__()
        self.mock_parameter = nn.Parameter(torch.tensor(1.)) # optimizer raises Value Error if no parameters exist
        self.z_dim = z_dim
    def forward(self, batch:torch.Tensor):
        Z = []
        for t, x in enumerate(torch.split(batch,1,1)):
            z_component = x[...,:self.z_dim]
            delta = Delta(z_component.squeeze(1))
            z = pyro.sample(f"z_{t + 1}", delta.to_event(1))
            Z.append(z)
        return Z


class TimeSeriesCNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_filters=64,sigma:float = 0.1):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=output_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(self, x):
        # x shape: [N_batches, N_time_steps, N_dimensions]
        x = x.permute(0, 2, 1)  # Change to [N_batches, N_dimensions, N_time_steps]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # Change back to [N_batches, N_time_steps, N_dimensions_output]

        Z = []
        for t, x in enumerate(torch.split(x, 1, 1)):
            normal = Normal(x.squeeze(1), torch.abs(self.sigma))
            z = pyro.sample(f"{V.LATENT}_{t + 1}", normal.to_event(1))
            Z.append(z)
        return Z



class SimpleNormalLatentFactorNNGuide(LightningModule):

    def __init__(self, n_time_steps: int, input_dim: int, n_latent_factors: int , hidden_dim):
        super().__init__()
        self.output_dim= n_latent_factors
        self.lin_1 = nn.Linear(n_time_steps*input_dim,hidden_dim)
        self.relu_1 = nn.ReLU()
        self.scale_layer = nn.Linear(hidden_dim, n_latent_factors ** 2)
        self.softplus= nn.Softplus()
        self.mu_layer = nn.Linear(hidden_dim,n_latent_factors)


    def forward(self, batch: torch.Tensor):
        x = self.relu_1(self.lin_1(batch.reshape(batch.size(0),-1 )))
        scale = self.scale_layer(x).reshape(batch.size(0),batch.size(1),self.output_dim,self.output_dim)
        scale_diag = torch.diagonal(scale,dim1=-2,dim2=-1)
        scale_diag = self.softplus(scale_diag)
        scale_diag = torch.diag_embed(scale_diag)
        scale_lower_tria = torch.tril(scale,diagonal=-1)
        scale = scale_lower_tria + scale_diag

        mu = self.mu_layer(x).reshape(batch.size(0),batch.size(1),self.output_dim)

        pyro.sample("latent_factors", MultivariateNormal(mu, scale_tril=scale).to_event(1))




