import torch

from src.lightning_modules.variational_lightning_module import LightningVariationalHiddenMarkov
from src.pyro_messengers.handlers import mean, force
from torch.nn.functional import mse_loss
from pyro.poutine.handlers import trace
class LightningMSEHiddenMarkov(LightningVariationalHiddenMarkov):


    def training_step(self, batch: torch.Tensor):
        guide_trace = trace(self.variational_distribution).get_trace(batch)
        with mean():
            with force(trace = guide_trace,forcing_interval=1, latent_group_name="latent", alpha=0.2,subspace_dim = 1):
                z_tensor, x_tensor = self.hidden_markov_model.get_history(batch)

        loss = mse_loss(x_tensor, batch)

        return loss






