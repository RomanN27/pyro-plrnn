import pyro
import torch
import tqdm
from hydra.utils import instantiate
from pyro import poutine as poutine, distributions as dist
from pyro.poutine import scale
from pyro.distributions.transforms import affine_autoregressive
from torch import nn as nn
from pyro.distributions import TransformedDistribution
from pyro.poutine import uncondition, trace

from dataloader import get_data_of_one_subject
from plrnns import LinearObservationModel, PLRNN, Combiner
from torch.utils.data import Dataset
from typing import Callable
from pyro.optim import PyroOptim
from pyro.infer import ELBO, SVI
from dataclasses import dataclass
import logging
import mlflow
from tqdm import tqdm
import mlflow
from pyro.infer import Trace_ELBO
from time_series_model import TimeSeriesModel


@dataclass
class TrainingConfig:
    annealing_factor: float = 1.0
    annealing_epochs: int = 3


class AnnealingTimeSeriesTrainer:

    def __init__(self, time_series_model: TimeSeriesModel, variational_distribution: Callable,
                 data_loader: DataSet, optimizer: PyroOptim, elbo: ELBO):

        self.time_series_model = time_series_model
        self.variational_distribution = variational_distribution
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.elbo = elbo



        self.svi = SVI(self.time_series_model, variational_distribution, self.optimizer, loss=self.elbo)

    def train(self, n_epochs: int, min_af: int, annealing_epochs: int):
        orig_sampler = self.time_series_model.sample_next_hidden_state

        step = 0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            annealing_factor = self.scale_hidden_state_sampler(annealing_epochs, epoch, min_af, orig_sampler)

            for batch in tqdm(self.data_loader):
                loss = self.svi.step(*batch, annealing_factor)
                epoch_loss += loss
                if epoch > 5:
                    mlflow.log_metric("loss", f"{loss:2f}", step=step)
                step += 1
            print(epoch_loss)

    def scale_hidden_state_sampler(self, annealing_epochs, epoch, min_af, orig_sampler):
        annealing_factor = min_af + (1 - min_af) * min(1, epoch / annealing_epochs)
        scaled_sampler = scale(orig_sampler, annealing_factor)
        self.time_series_model.sample_next_hidden_state = scaled_sampler
        return annealing_factor

    def save_checkpoint(self):
        pass


def get_trainer_from_config(cfg):
    data = instantiate(cfg.data)
    plrnn = instantiate(cfg.transition_model)
    observation_model = instantiate(cfg.observation_model)
    observation_distribution = instantiate(cfg.observation_distribution)
    transition_distribution = instantiate(cfg.transition_distribution)
    time_series_model = TimeSeriesModel(plrnn, observation_model, observation_distribution, transition_distribution,)

    optimizer_class = instantiate(cfg.optimizer.optimizer_class)
    optimizer = optimizer_class(dict(cfg.optimizer.optim_args))
    guide = instantiate(cfg.guide)
    loss = instantiate(cfg.loss)
    trainer = AnnealingTimeSeriesTrainer(time_series_model, guide, data, optimizer, loss)
    return trainer
