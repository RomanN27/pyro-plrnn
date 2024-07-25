import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import TypeVar, Protocol, Callable, Type, Any

import mlflow
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, LightningDataModule
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pyro.infer import Predictive
from pyro.optim import PyroOptim
from pyro.poutine.messenger import Messenger
from torch.utils.data import Dataset
from pyro.poutine.trace_messenger import TraceMessenger
from src.metrics.metrics import PyroTimeSeriesMetricCollection
from src.models.forecaster import Forecaster
from src.models.hidden_markov_model import HiddenMarkovModel
from src.training.messengers.handlers import observe
from src.utils.custom_typehint import TensorIterable
from src.constants import OBSERVED_VARIABLE_NAME as X
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
T = TypeVar("T", bound="TimeSeriesModule")


class ElboLoss(Protocol):

    def differentiable_loss(self, model: Callable, guide: Callable, *args, **kwargs) -> torch.Tensor: ...


class LightningVariationalHiddenMarkov(LightningModule):
    def __init__(self, hidden_markov_model: HiddenMarkovModel, variational_distribution: nn.Module,
                 data_loader: LightningDataModule, optimizer: Type[Optimizer], loss: ElboLoss,
                 metric_collection: PyroTimeSeriesMetricCollection,
                 messengers: Messenger, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.hidden_markov_model = hidden_markov_model
        self.variational_distribution = variational_distribution
        self.data_loader = data_loader
        self.optimizer_cls = optimizer
        self.loss = loss

        self.predictive = Predictive(self.hidden_markov_model, guide=self.variational_distribution, num_samples=1)
        self.metric_collection = metric_collection
        self.forecaster = Forecaster(self.hidden_markov_model, self.variational_distribution)
        self.messengers = [messengers]


    def configure_optimizers(self):
        #return self.optimizer_cls
        hhm_parameters = list(self.hidden_markov_model.parameters())
        #sigma_parameters = hhm_parameters.pop(-2)
        #bias_parameters = hhm_parameters.pop(1)
        vae_parameters = list(self.variational_distribution.parameters())

        hmm_optimizer = self.optimizer_cls([
            {"params": hhm_parameters},
            #{"params":sigma_parameters, "lr":0.1},
            #{"params": bias_parameters, "lr": 0.1}
        ])

        vae_optimizer = self.optimizer_cls(vae_parameters)
        var_lr_scheduler = ReduceLROnPlateau(vae_optimizer,patience=4)
        return [hmm_optimizer, vae_optimizer], [var_lr_scheduler]


    # def validation_step(self, batch: TensorIterable) -> None:
    #     self.metric_collection.update(self.forecaster, batch)
    #     results = self.metric_collection.compute()
    #     results = {k: v.numpy() for k, v in results.items()}
    #     #self.logger.log_metrics(results)
    #     self.metric_collection.reset()

    def training_step(self, batch: torch.Tensor):
        #self.loss.alpha = min(self.loss.alpha*2 if not self.current_epoch % 5 and self.current_epoch > 0 else self.loss.alpha,1)

        hmm_optimizer, vae_optimizer = self.optimizers()
        var_lr_scheduler = self.lr_schedulers()
        hmm_optimizer.zero_grad()
        vae_optimizer.zero_grad()

        #with ExitStack() as stack:
        #    for msgr in self.messengers:
        #        stack.enter_context(msgr)

        with observe(batch=batch, observation_group_symbol=X):
            loss, vanilla, dsr = self.loss.differentiable_loss(self.hidden_markov_model, self.variational_distribution, batch)


        self.metric_collection.update(batch,self.hidden_markov_model,self.variational_distribution)
        self.metric_collection.plot()

        vanilla.backward()
        vae_optimizer.step()

        hmm_optimizer.zero_grad()

        dsr.backward()
        hmm_optimizer.step()
        var_lr_scheduler.step(dsr)

        self.log("dsr_loss", dsr,prog_bar=True,on_step=True,on_epoch=True)
        self.log("vanilla_loss", dsr, prog_bar=True, on_step=True,on_epoch=True)
        self.log("loss", loss, prog_bar=True,on_step=True, on_epoch=True)
        self.log("alpha", self.loss.alpha, prog_bar=True, on_step=True, on_epoch=False)

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
            "hidden_markov_model": self.hidden_markov_model.state_dict(),
            "variational_distribution": self.variational_distribution.state_dict(),
           # "optimizer": self.optimizers().state_dict()

        }, path)

    def load(self, path: Path | str, just_try=True):
        checkpoint = torch.load(path)
        try:
            self.hidden_markov_model.load_state_dict(checkpoint["hidden_markov_model"])
        except Exception as e:
            if not just_try:
                raise e
            else:
                logging.warning("Loading didn'delta_t work for time series model")

        try:
            self.variational_distribution.load_state_dict(checkpoint["variational_distribution"])
        except Exception as e:
            if not just_try:
                raise e
            else:
                logging.warning("Loading didn'delta_t work for variational_distribution")

        try:
            self.optimizer_cls.set_state(checkpoint["optimizer"])
        except Exception as e:
            if not just_try:
                raise e
            else:
                logging.warning("Loading didn'delta_t work for optimizer_cls")
