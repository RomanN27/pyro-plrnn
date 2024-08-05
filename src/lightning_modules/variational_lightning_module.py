from typing import TypeVar, Generic

import torch
from lightning.pytorch.utilities import grad_norm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.lightning_modules.base_lightning_module import BaseLightninglHiddenMarkov, Stage
from src.metrics.metric_base import Logger

from src.pyro_messengers.handlers import observe
from src.utils.variable_group_enum import V
from src.training.losses import TeacherForcingTraceELBO

from src.data.data_module import DataType, DatasetType

from src.models.hidden_markov_model import LatentModelType, ObservationModelType
T = TypeVar("T", bound="TimeSeriesModule")


class VariationalTeacherForcing(BaseLightninglHiddenMarkov,
                                Generic[DatasetType, DataType, LatentModelType, ObservationModelType]):

    logger: Logger

    def __init__(self,forcing_interval: int, alpha: float, subspace_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forcing_interval = forcing_interval
        self.loss = TeacherForcingTraceELBO(forcing_interval,alpha,subspace_dim)

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

        self.update_metric_collection(Stage.train, batch)





