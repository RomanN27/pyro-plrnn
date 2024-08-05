from enum import StrEnum
from typing import Type, Optional, Protocol, Callable

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pyro.poutine.messenger import Messenger
from torch import nn as nn
from torch.optim import Optimizer

from src.metrics.metric_base import MetricCollection
from src.models.forecaster import Forecaster
from src.models.hidden_markov_model import HiddenMarkovModel, ObservationModelType, LatentModelType


class Stage(StrEnum):
    train = "train"
    val = "val"
    test = "test"


class ElboLoss(Protocol):

    def differentiable_loss(self, model: Callable, guide: Callable, *args, **kwargs) -> torch.Tensor: ...

class BaseLightninglHiddenMarkov(LightningModule):
    def __init__(self, hidden_markov_model: HiddenMarkovModel[ObservationModelType, LatentModelType],
                 variational_distribution: nn.Module,
                 optimizer: Type[Optimizer],
                 metric_collections: dict[Stage, MetricCollection],
                 messengers:Optional[ list[Messenger]]= None) -> None:
        super().__init__()

        self.hidden_markov_model = hidden_markov_model
        self.variational_distribution = variational_distribution
        self.optimizer_cls = optimizer

        self.metric_collections = torch.nn.ModuleDict(metric_collections)

        self.forecaster = Forecaster(self.hidden_markov_model, self.variational_distribution)
        self.messengers = messengers if messengers is not None else []

    def log_metric_collection(self, stage: Stage):
        metric_collection = self.metric_collections.get(stage)
        if metric_collection is not None:
            metric_collection.log(self.logger, _step = str(self.current_epoch))
            metric_collection.reset()

    def on_train_epoch_end(self) -> None:
        self.log_metric_collection(Stage.train)

    def update_metric_collection(self, stage: Stage, batch: torch.Tensor):
        metric_collection = self.metric_collections.get(stage)
        if metric_collection is not None:
            metric_collection.update(hmm = self.hidden_markov_model,
                                     batch = batch,
                                     forecaster = self.forecaster,
                                     guide = self.variational_distribution)

    def validation_step(self,batch: torch.Tensor) -> STEP_OUTPUT:
        self.update_metric_collection(Stage.val, batch)

    def on_validation_end(self) -> None:
        self.log_metric_collection(Stage.val)

    def test_step(self,batch: torch.Tensor) -> STEP_OUTPUT:
        torch.set_grad_enabled(True)
        self.update_metric_collection(Stage.test, batch)

    def on_test_end(self) -> None:
        self.log_metric_collection(Stage.test)


