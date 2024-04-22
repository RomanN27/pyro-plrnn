import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Any

import mlflow
import torch
import torch.nn as nn
import tqdm
from hydra.utils import instantiate
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from lightning.pytorch import Trainer

from pyro.infer import Predictive, Trace_ELBO
from pyro.optim import PyroOptim
from pyro.poutine import scale, block
from torch.utils.data import Dataset
from tqdm import tqdm

from time_series_model import TimeSeriesModel

if TYPE_CHECKING:
    from pyro.poutine.runtime import Message
from omegaconf import DictConfig
from pathlib import Path
from omegaconf import OmegaConf
from evaluation.metrics import PyroTimeSeriesMetricCollection
from lightning.pytorch import LightningModule
from custom_typehint import TensorIterable
from torch import Tensor
from evaluation.metrics import  PyroTimeSeriesMetricCollection
from forecaster import Forecaster
@dataclass
class TrainingConfig:
    annealing_factor: float = 1.0
    annealing_epochs: int = 3


T = TypeVar("T", bound="TimeSeriesModule")


class TimeSeriesModule(LightningModule):
    def __init__(self, time_series_model: TimeSeriesModel, variational_distribution: nn.Module,
                 data_loader: Dataset, optimizer: PyroOptim, elbo: Trace_ELBO, metric_collection: PyroTimeSeriesMetricCollection):
        super().__init__()
        self.time_series_model = time_series_model
        self.variational_distribution = variational_distribution
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.elbo = elbo
        self.predictive = Predictive(self.time_series_model, guide = self.variational_distribution,num_samples=1)
        self.metric_collection = metric_collection
        self.forecaster = Forecaster(self.time_series_model, self.variational_distribution)

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
        time_series_model = TimeSeriesModel(plrnn, observation_model, observation_distribution, transition_distribution)
        metrics = [instantiate(metric) for metric in cfg.metriccollection.metrics.values()]
        metric_collection = PyroTimeSeriesMetricCollection(metrics,**cfg.metriccollection.kwargs)
        optimizer_class = instantiate(cfg.optimizer.optimizer_class)
        optimizer = optimizer_class(dict(cfg.optimizer.optim_args))
        guide = instantiate(cfg.guide)
        loss = instantiate(cfg.loss)
        trainer = cls(**cfg.module,time_series_model = time_series_model,variational_distribution= guide,
                      data_loader = data_module, optimizer=optimizer, elbo= loss,metric_collection = metric_collection)
        return trainer

    def validation_step(self, batch: TensorIterable) -> None:
        self.metric_collection.update(self.forecaster,batch)
        results = self.metric_collection.compute()
        self.logger.log_metrics(results)

    @classmethod
    def get_config_from_run_id(cls: type[T], run_id: str) -> DictConfig:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
        run_data_dict = mlflow_client.get_run(run_id=run_id).data.to_dictionary()
        return OmegaConf.create(run_data_dict["params"]["config"])

    @classmethod
    def get_trainer_from_run_id_config(cls, run_id: str):
        return cls.get_trainer_from_config(cls.get_config_from_run_id(run_id))

    def save(self,path: Path | str):
        torch.save({
            "time_series_model": self.time_series_model.state_dict(),
            "variational_distribution": self.variational_distribution.state_dict(),
            "optimizer": self.optimizer.get_state()

        },path )

    def load(self,path:Path | str,just_try = True):
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




class AnnealingModule(TimeSeriesModule):


    def __init__(self, beginning_annealing_factor: int, annealing_epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_annealing_factor = beginning_annealing_factor
        self.delta = (1 - beginning_annealing_factor) / annealing_epochs

    @property
    def current_annealing_factor(self):
        return self._current_annealing_factor

    @current_annealing_factor.setter
    def current_annealing_factor(self,value):
        #clip annealing factor to allowed range in [0,1]
        self._current_annealing_factor = max(min(1, value),0)
    def increase_annealing_factor(self):
        self.current_annealing_factor += self.delta

    def annealing_selector(self, msg: "Message") -> bool:
        z_name = self.time_series_model.HIDDEN_VARIABLE_NAME
        pattern = rf"^{z_name}_\d+"
        name = msg["name"]
        return name and bool(re.match(pattern, name))

    def annealing_hider(self,msg: "Message") -> bool:
        return not self.annealing_selector(msg)
    def training_step(self, batch: TensorIterable):

        with scale(scale = self.current_annealing_factor):
            with block(hide_fn=self.annealing_hider):
                    loss = self.elbo.differentiable_loss(self.time_series_model,self.variational_distribution, batch)

        self.logger.experiment.log_metric("loss", f"{loss / 10000:2f}", step=self.global_step)
    def on_train_epoch_end(self) -> None:
        self.increase_annealing_factor()




