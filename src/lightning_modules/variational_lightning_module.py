from enum import StrEnum
from typing import TypeVar, Protocol, Callable, Type, Generic, TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pyro.poutine.messenger import Messenger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.metrics.metric_base import MetricCollection, Logger

from src.models.forecaster import Forecaster
from src.models.hidden_markov_model import HiddenMarkovModel
from src.pyro_messengers.handlers import observe
from src.utils.variable_group_enum import V


from src.data.time_series_dataset import TimeSeriesChunkDataset
from src.data.data_module import DataType, DatasetType, DataLoader

from src.models.hidden_markov_model import LatentModelType, ObservationModelType
T = TypeVar("T", bound="TimeSeriesModule")


class Stage(StrEnum):
    train = "train"
    val = "val"
    test = "test"


class ElboLoss(Protocol):

    def differentiable_loss(self, model: Callable, guide: Callable, *args, **kwargs) -> torch.Tensor: ...


class LightningVariationalHiddenMarkov(LightningModule,
                                       Generic[DatasetType, DataType, LatentModelType, ObservationModelType]):

    logger: Logger
    def __init__(self, hidden_markov_model: HiddenMarkovModel[ObservationModelType, LatentModelType],
                 variational_distribution: nn.Module,
                 optimizer: Type[Optimizer], loss: ElboLoss,
                 metric_collections: dict[Stage, MetricCollection],
                 messengers: Messenger, **kwargs):
        super().__init__()

        self.automatic_optimization = False
        self.hidden_markov_model = hidden_markov_model
        self.variational_distribution = variational_distribution
        self.optimizer_cls = optimizer
        self.loss = loss

        self.metric_collections = metric_collections
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
        var_lr_scheduler = ReduceLROnPlateau(vae_optimizer, patience=4)
        return [hmm_optimizer, vae_optimizer], [var_lr_scheduler]

    # def validation_step(self, batch: TensorIterable) -> None:
    #     self.metric_collections.update(self.forecaster, batch)
    #     results = self.metric_collections.compute()
    #     results = {k: v.numpy() for k, v in results.items()}
    #     #self.logger.log_metrics(results)
    #     self.metric_collections.reset()

    def log_grads(self, grad_name: str):
        grads = grad_norm(self, 2)
        self.logger.experiment.log_dict(self.logger.run_id, {k: v.tolist() for k, v in grads.items()},
                                        f"grads/{self.current_epoch}/{grad_name}_grads.json")

    def training_step(self, batch: torch.Tensor):
        #self.loss.alpha = min(self.loss.alpha*2 if not self.current_epoch % 5 and self.current_epoch > 0 else self.loss.alpha,1)

        hmm_optimizer, vae_optimizer = self.optimizers()


        var_lr_scheduler = self.lr_schedulers()
        hmm_optimizer.zero_grad()
        vae_optimizer.zero_grad()

        #with ExitStack() as stack:
        #    for msgr in self.pyro_messengers:
        #        stack.enter_context(msgr)

        with observe(batch=batch, observation_group_symbol=V.OBSERVED):
            loss, vanilla, dsr = self.loss.differentiable_loss(self.hidden_markov_model, self.variational_distribution,batch)

        normalization_factor = len(batch.reshape(-1))
        loss, vanilla, dsr = loss / normalization_factor, vanilla / normalization_factor, dsr / normalization_factor

        vae_optimizer: Optimizer
        a = vae_optimizer.state_dict()

        vanilla.backward()

        self.log_grads("vanilla")
        vae_optimizer.step()

        #hmm_optimizer.zero_grad()

        dsr.backward()
        self.log_grads("dsr")
        hmm_optimizer.step()
        var_lr_scheduler.step(dsr)

        self.log("dsr_loss", dsr, prog_bar=True, on_step=False, on_epoch=True)
        self.log("vanilla_loss", vanilla, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        self.log_metric_collection(Stage.train)
    def log_metric_collection(self, stage: Stage):
        metric_collection = self.metric_collections.get(stage)
        if metric_collection is not None:
            metric_collection.log(self.logger, _step = str(self.current_epoch))
            metric_collection.reset()

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





