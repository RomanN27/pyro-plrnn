import pyro
import torch
from pyro import poutine as poutine, distributions as dist
from utils import pad_and_reverse
from pyro.distributions.transforms import affine_autoregressive
from torch import nn as nn
from pyro.distributions import TransformedDistribution
from plrnns import LinearObservationModel, PLRNN, Combiner
from torch.utils.data import DataLoader
from typing import Callable
from pyro.optim import PyroOptim
from pyro.infer import ELBO
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyro.distributions import TorchDistributionMixin

class RNNGuide(nn.Module):

    def __init__(self, rnn:nn.RNN, combiner: Combiner, dist: "TorchDistributionMixin" ):
        super().__init__()
        self.rnn = rnn
        self.combiner = combiner
        self.dist = dist

        self.z_q_0 = nn.Parameter(torch.zeros(self.combiner.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn.hidden_size))

    def __call__(self,
                 batch,
                 batch_reversed,
                 batch_mask,
                 batch_seq_lengths,
                 annealing_factor
                 ):
        pyro.module("dmm", self)
        T_max =  batch.size(1)
        h_0_contig = self.h_0.expand(
            self.rnn.num_layers, batch.size(0), self.rnn.hidden_size
        ).contiguous()

        rnn_output, _ = self.rnn(batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = pad_and_reverse(rnn_output, batch_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(batch)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = self.dist(z_loc,z_scale)


                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):

                    z_t = pyro.sample(
                        "z_%d" % t, z_dist.mask(batch_mask[:, t - 1:t]).to_event(1)
                    )

                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t