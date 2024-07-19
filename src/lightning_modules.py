import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import TypeVar, Protocol, Callable

import mlflow
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pyro.infer import Predictive
from pyro.optim import PyroOptim
from pyro.poutine.messenger import Messenger
from torch.utils.data import Dataset

from src.metrics.metrics import PyroTimeSeriesMetricCollection
from src.models.forecaster import Forecaster
from src.models.time_series_model import HiddenMarkovModel
from src.training.messengers.handlers import observe
from src.utils.custom_typehint import TensorIterable

T = TypeVar("T", bound="TimeSeriesModule")


class ElboLoss(Protocol):

    def differentiable_loss(self, model: Callable, guide: Callable, *args, **kwargs) -> torch.Tensor: ...


class LightningVariationalHiddenMarkov(LightningModule):
    def __init__(self, time_series_model: HiddenMarkovModel, variational_distribution: nn.Module,
                 data_loader: Dataset, optimizer: PyroOptim, elbo: ElboLoss,
                 metric_collection: PyroTimeSeriesMetricCollection,
                 messenger: list[Messenger]):
        super().__init__()
        self.time_series_model = time_series_model
        self.variational_distribution = variational_distribution
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.elbo = elbo
        self.predictive = Predictive(self.time_series_model, guide=self.variational_distribution, num_samples=1)
        self.metric_collection = metric_collection
        self.forecaster = Forecaster(self.time_series_model, self.variational_distribution)
        self.messenger = messenger

    def configure_optimizers(self):
        #return self.optimizer
        parameters = list(self.time_series_model.parameters()) + list(self.variational_distribution.parameters())
        return torch.optim.Adam(parameters, lr=0.01)

    @classmethod
    def get_trainer_from_config(cls: type[T], cfg: DictConfig) -> T:
        #TODO Use Lightning CLI
        data_module = instantiate(cfg.data)
        plrnn = instantiate(cfg.transition_model)
        observation_model = instantiate(cfg.observation_model)
        observation_distribution = instantiate(cfg.observation_distribution)
        transition_distribution = instantiate(cfg.transition_distribution)
        time_series_model = HiddenMarkovModel(plrnn, observation_model, observation_distribution,
                                              transition_distribution)
        metrics = [instantiate(metric) for metric in cfg.metriccollection.metrics.values()]
        metric_collection = PyroTimeSeriesMetricCollection(metrics, **cfg.metriccollection.kwargs)
        optimizer_class = instantiate(cfg.optimizer.optimizer_class)
        optimizer = optimizer_class(dict(cfg.optimizer.optim_args))
        guide = instantiate(cfg.guide)
        loss = instantiate(cfg.loss)
        trainer = cls(**cfg.module, time_series_model=time_series_model, variational_distribution=guide,
                      data_loader=data_module, optimizer=optimizer, elbo=loss, metric_collection=metric_collection)
        return trainer

    def validation_step(self, batch: TensorIterable) -> None:
        self.metric_collection.update(self.forecaster, batch)
        results = self.metric_collection.compute()
        results = {k: v.numpy() for k, v in results.items()}
        #self.logger.log_metrics(results)
        self.metric_collection.reset()

    def training_step(self, batch: torch.Tensor):
        with ExitStack() as stack:
            for msgr in self.messenger:
                stack.enter_context(msgr)

            with observe(batch=batch, observation_group_symbol=self.time_series_model.OBSERVED_VARIABLE_NAME):
                loss = self.elbo.differentiable_loss(self.time_series_model, self.guide, batch)

        return loss

    @classmethod
    def get_config_from_run_id(cls: type[T], run_id: str) -> DictConfig:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
        run_data_dict = mlflow_client.get_run(run_id=run_id).data.to_dictionary()
        return OmegaConf.create(run_data_dict["params"]["config"])

    @classmethod
    def get_trainer_from_run_id_config(cls, run_id: str):
        return cls.get_trainer_from_config(cls.get_config_from_run_id(run_id))

    def save(self, path: Path | str):
        torch.save({
            "time_series_model": self.time_series_model.state_dict(),
            "variational_distribution": self.variational_distribution.state_dict(),
            "optimizer": self.optimizer.get_state()

        }, path)

    def load(self, path: Path | str, just_try=True):
        checkpoint = torch.load(path)
        try:
            self.time_series_model.load_state_dict(checkpoint["time_series_model"])
        except Exception as e:
            if not just_try:
                raise e
            else:
                logging.warning("Loading didn't work for time series model")

        try:
            self.variational_distribution.load_state_dict(checkpoint["variational_distribution"])
        except Exception as e:
            if not just_try:
                raise e
            else:
                logging.warning("Loading didn't work for variational_distribution")

        try:
            self.optimizer.set_state(checkpoint["optimizer"])
        except Exception as e:
            if not just_try:
                raise e
            else:
                logging.warning("Loading didn't work for optimizer")
