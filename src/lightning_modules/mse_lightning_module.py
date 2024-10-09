import torch
from typing import  Optional

from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.lightning_modules.base_lightning_module import BaseLightninglHiddenMarkov
from src.lightning_modules.base_lightning_module import Stage
from src.pyro_messengers.handlers import mean, force
from src.utils.variable_group_enum import V
from torch.nn.functional import mse_loss
from src.regularization.manifold_attractor_regularization import ManifoldAttractorRegularization
from pyro.poutine.handlers import trace
import  matplotlib.pyplot as plt
from lightning.pytorch.utilities import grad_norm
plt.ioff()
class MSETeacherForcing(BaseLightninglHiddenMarkov):

    def __init__(self,forcing_interval: int, alpha: float, subspace_dim: Optional[int] = None,
                 lambda_ =  1, n_target_points =-1, warm_start:int =  50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.forcing_interval = forcing_interval
        self.alpha = alpha
        self.subspace_dim = subspace_dim
        self.reg = ManifoldAttractorRegularization(lambda_, n_target_points)
        self.warm_start = warm_start

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        return self.optimizer_cls(self.hidden_markov_model.parameters())

    def training_step(self, batch: torch.Tensor):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        guide_trace = trace(self.variational_distribution).get_trace(batch)
        with mean():
            with force(trace = guide_trace,forcing_interval=self.forcing_interval, latent_group_name=V.LATENT,
                       alpha=self.alpha,subspace_dim=self.subspace_dim):
                z_tensor, x_tensor = self.hidden_markov_model.get_history(batch)

        loss = mse_loss(x_tensor, batch)
        reg = self.reg(self.hidden_markov_model.transition_sampler.model)
        self.log("mse_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("reg_loss", reg, prog_bar=True, on_step=True, on_epoch=True)
        self.update_metric_collection(Stage.train, batch)
        self.manual_backward(loss + reg)
        self.on_before_optimizer_step(optimizer.optimizer)
        optimizer.step()
        return loss + reg


class HierarchicalMSETeacherForcing(BaseLightninglHiddenMarkov):

    def configure_optimizers(self):
        hmm_parameters = self.hidden_markov_model.named_parameters()






