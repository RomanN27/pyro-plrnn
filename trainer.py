import pyro
import torch
import tqdm
from hydra.utils import instantiate
from pyro import poutine as poutine, distributions as dist

from pyro.distributions.transforms import affine_autoregressive
from torch import nn as nn
from pyro.distributions import TransformedDistribution
from pyro.poutine import uncondition, trace

from dataloader import get_data_of_one_subject
from plrnns import LinearObservationModel, PLRNN, Combiner
from torch.utils.data import DataLoader
from typing import Callable
from pyro.optim import PyroOptim
from pyro.infer import ELBO, SVI
from dataclasses import dataclass
import logging
import mlflow
from tqdm import tqdm
import mlflow
from pyro.infer import  Trace_ELBO

@dataclass
class TrainingConfig:
    annealing_factor: float  = 1.0
    annealing_epochs: int = 3

class PLRNNTrainer(nn.Module):
    HIDDEN_VARIABLE_NAME = "z"
    OBSERVED_VARIABLE_NAME = "x"

    def __init__(self, transition_model: PLRNN, observation_model: nn.Module ,variational_distribution: Callable,
                 data_loader:DataLoader, optimizer: PyroOptim, elbo: ELBO):
        super().__init__()
        self.transition_model = transition_model
        self.observation_model = observation_model
        #make sure to register nn.Module that are used in the variational distribution (aka guide)
        self.variational_distribution = variational_distribution
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.elbo = elbo

        self.z_0 = nn.Parameter(torch.zeros(self.transition_model.z_dim))

        self.svi = SVI(self.time_series_model, variational_distribution, self.optimizer, loss = self.elbo)


    def train(self,n_epochs: int, min_af:int , annealing_epochs: int):
        step = 0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            annealing_factor = min_af + (1 - min_af) * min(1, epoch / annealing_epochs)
            for batch in tqdm(self.data_loader):
                loss = self.svi.step(*batch,annealing_factor)
                epoch_loss+= loss
                if epoch > 5:
                    mlflow.log_metric("loss", f"{loss:2f}", step=step)
                step+=1
            print(epoch_loss)


    def save_checkpoint(self):
        pass

    def time_series_model(self,
                          batch,
                          batch_reversed,
                          batch_mask,
                          batch_seq_lengths,annealing_factor=1.0):
        pyro.module("dmm", self)

        T_max = batch.size(1)
        n_batches = batch.size(0)

        z_prev = self.z_0.repeat(n_batches,1)

        with pyro.plate("z_minibatch", n_batches):

            for t in pyro.markov(range(1, T_max + 1)):

                z_loc, z_scale = self.transition_model(z_prev)

                z_t = self.sample_next_hidden_state(batch_mask, t, z_loc, z_scale,annealing_factor)

                observation_mean, observation_covariance = self.observation_model(z_t)

                self.sample_observation(batch, batch_mask, observation_covariance, observation_mean, t)

                z_prev = z_t


    def sample_observed_time_series(self):
        unconditioned_model = trace(uncondition(self.time_series_model))
        #batch doesnt matter, only for shape infering
        batch = next(iter(self.data_loader))

        unconditioned_model(*batch)

        filter_observations = lambda pair: pair[0].startswith(self.OBSERVED_VARIABLE_NAME + "_")

        obs_nodes = dict(filter(filter_observations, unconditioned_model.msngr.trace.nodes.items()))
        values = [x["value"] for x in obs_nodes.values()]
        time_series = torch.stack(values, 1)
        return time_series


    def sample_observation(self, batch, batch_mask, observation_covariance, observation_mean, t):
        pyro.sample(
            f"{self.OBSERVED_VARIABLE_NAME}_{t}",
            dist.Normal(observation_mean, observation_covariance)
            .mask(batch_mask[:, t - 1: t])
            .to_event(1),
            obs=batch[:, t - 1, :],
        )

    def sample_next_hidden_state(self, batch_mask, t, z_loc, z_scale,annealing_factor):
        with poutine.scale(scale = annealing_factor):
            z_t = pyro.sample(
                f"{self.HIDDEN_VARIABLE_NAME}_{t}",
                dist.Normal(z_loc, z_scale)
                .mask(batch_mask[:, t - 1: t])
                .to_event(1),
            )
        return z_t



def get_trainer_from_config(cfg):
    train, test, val = get_data_of_one_subject(cfg.data.subject_index)
    plrnn = instantiate(cfg.transition_model)
    observation_model = instantiate(cfg.observation_model)
    optimizer_class = instantiate(cfg.optimizer.optimizer_class)
    optimizer = optimizer_class(dict(cfg.optimizer.optim_args))
    guide = instantiate(cfg.guide)
    loss = instantiate(cfg.loss)
    trainer = PLRNNTrainer(plrnn, observation_model, guide, train, optimizer, loss)
    return trainer
