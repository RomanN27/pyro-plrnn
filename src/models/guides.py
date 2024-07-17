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
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
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
    def __init__(self, rnn: nn.RNN, combiner: Combiner, initial_state_guide: InitialStateGuide, dist: Type["TorchDistributionMixin"]):
        super().__init__()
        self.rnn = rnn
        self.combiner = combiner
        self.initial_state_guide = initial_state_guide
        self.dist = dist
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn.hidden_size))

    def __call__(self, batch: TensorIterable):
        padded_sequence, batch_reversed, batch_mask, batch_seq_lengths = collate_fn(batch)
        pyro.module("dmm", self)
        T_max = padded_sequence.size(1)
        n = padded_sequence.size(0)
        h_0_contig = self.h_0.expand(self.rnn.num_layers, n, self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(batch_reversed, h_0_contig)
        rnn_output = pad_and_reverse(rnn_output, batch_seq_lengths)

        # Infer initial state z_q_0
        x_0 = padded_sequence[:, 0, :]  # Assuming the input x is available in the first time step
        locs, scales, mixing_weights = self.initial_state_guide(x_0)

        z_q_0 = pyro.sample("z_0", dist.MixtureSameFamily(
            dist.Categorical(mixing_weights),
            dist.Independent(dist.Normal(locs, scales), 1)
        ).to_event(1))

        z_prev = z_q_0
        for t in pyro.markov(range(1, T_max + 1)):
            z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
            z_dist = self.dist(z_loc, z_scale)
            masked_distribution = z_dist.mask(batch_mask[:, t - 1:t])
            z_t = pyro.sample("z_%d" % t, masked_distribution.to_event(1))
            z_prev = z_t
