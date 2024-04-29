import pyro
import torch
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
    def ___call__(self,batch):
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




class RNNGuide(LightningModule):

    def __init__(self, rnn:nn.RNN, combiner: Combiner, dist: Type["TorchDistributionMixin"] ):
        super().__init__()
        self.rnn = rnn
        self.combiner = combiner
        self.dist = dist

        self.z_q_0 = nn.Parameter(torch.zeros(self.combiner.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn.hidden_size))

    def __call__(self, batch: TensorIterable):
        padded_sequence, batch_reversed, batch_mask, batch_seq_lengths = collate_fn(batch)
        pyro.module("dmm", self)
        T_max =  padded_sequence.size(1)

        n = padded_sequence.size(0)
        h_0_contig = self.h_0.expand(
            self.rnn.num_layers, n, self.rnn.hidden_size
        ).contiguous()

        rnn_output, _ = self.rnn(batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = pad_and_reverse(rnn_output, batch_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(n, self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.

        # sample the latents z one time step at a time
        # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
        for t in pyro.markov(range(1, T_max + 1)):
            # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
            z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

            z_dist = self.dist(z_loc,z_scale)


            # sample z_t from the distribution z_dist

            masked_distribution = z_dist.mask(batch_mask[:, t - 1:t])
            z_t = pyro.sample(
                "z_%d" % t, masked_distribution.to_event(1)
            )

            # the latent sampled at this time step will be conditioned upon in the next time step
            # so keep track of it
            z_prev = z_t


