import logging
import os

import tqdm
from hydra.utils import instantiate
from pyro.poutine import scale, block
from torch.utils.data import Dataset
from pyro.optim import PyroOptim
from pyro.infer import ELBO, SVI
from dataclasses import dataclass
from tqdm import tqdm
import mlflow
import re
import torch.nn as nn
from src.models.hidden_markov_model import HiddenMarkovModel
from typing import TYPE_CHECKING, TypeVar
import torch
if TYPE_CHECKING:
    from pyro.poutine.runtime import Message
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import OmegaConf
from src.metrics.metrics import PyroTimeSeriesMetricCollection
@dataclass
class TrainingConfig:
    annealing_factor: float = 1.0
    annealing_epochs: int = 3


T = TypeVar("T", bound="Trainer")


class Trainer(ABC):
    ML_FLOW_ARTIFACTS_SAVING_PATH = "../mlartifacts"
    def __init__(self, time_series_model: HiddenMarkovModel, variational_distribution: nn.Module,
                 data_loader: Dataset, optimizer: PyroOptim, elbo: ELBO):
        self.time_series_model = time_series_model
        self.variational_distribution = variational_distribution
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.elbo = elbo

        self.svi = SVI(self.time_series_model, variational_distribution, self.optimizer, loss=self.elbo)

    @abstractmethod
    def train(self,*args,**kwargs):
        pass

    @classmethod
    def get_trainer_from_config(cls: type[T], cfg: DictConfig) -> T:
        train, test, valid = instantiate(cfg.data)
        plrnn = instantiate(cfg.transition_model)
        observation_model = instantiate(cfg.observation_model)
        observation_distribution = instantiate(cfg.observation_distribution)
        transition_distribution = instantiate(cfg.transition_distribution)
        time_series_model = HiddenMarkovModel(plrnn, observation_model, observation_distribution, transition_distribution)
        metrics = [instantiate(metric) for metric in cfg.metriccollection.metrics.values()]
        metric_collection = PyroTimeSeriesMetricCollection(metrics,**cfg.metriccollection.kwargs)
        optimizer_class = instantiate(cfg.optimizer.optimizer_class)
        optimizer = optimizer_class(dict(cfg.optimizer.optim_args))
        guide = instantiate(cfg.guide)
        loss = instantiate(cfg.loss)
        trainer = cls(time_series_model, guide, train, optimizer, loss)
        return trainer


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
            "hidden_markov_model": self.time_series_model.state_dict(),
            "variational_distribution": self.variational_distribution.state_dict(),
            "optimizer_cls": self.optimizer.get_state()

        },path )

    def load(self,path:Path | str,just_try = True):
        checkpoint = torch.load(path)
        try:
            self.time_series_model.load_state_dict(checkpoint["hidden_markov_model"])
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
            self.optimizer.set_state(checkpoint["optimizer_cls"])
        except Exception as e:
            if not just_try:
                raise e
            else:
                logging.warning("Loading didn'delta_t work for optimizer_cls")




class AnnealingTrainer(Trainer):

    def annealing_selector(self, msg: "Message") -> bool:
        z_name = self.time_series_model.HIDDEN_VARIABLE_NAME
        pattern = rf"^{z_name}_\d+"
        name = msg["name"]
        return name and bool(re.match(pattern, name))

    def annealing_hider(self,msg: "Message") -> bool:
        return not self.annealing_selector(msg)
    def train(self, n_epochs: int, annealing_factor: int, annealing_epochs: int):

        step = 0
        min_loss = None
        last_epoch_loss = 0

        for epoch in (tbar := tqdm(range(n_epochs))):
            epoch_loss = 0.0
            annealing_factor = self.get_annealing_factor(annealing_epochs, epoch, annealing_factor)

            with scale(scale = annealing_factor):
                with block(hide_fn=self.annealing_hider):
                    for batch in self.data_loader:
                        loss = self.svi.step(batch)
                        dict_ = {"last_epoch_loss": last_epoch_loss, "batch_loss": loss, "min_loss": min_loss}
                        tbar.set_postfix(dict_)
                        epoch_loss += loss
                        if epoch > 5:
                            mlflow.log_metric(ö)

                        step += 1
                        last_epoch_loss = epoch_loss
            min_loss = min(min_loss, epoch_loss) if min_loss else epoch_loss

    def get_annealing_factor(self, annealing_epochs, epoch, min_af) -> float:
        annealing_factor = min_af + (1 - min_af) * min(1, epoch / annealing_epochs)
        return annealing_factor

