import torch

from src.lightning_modules.variational_lightning_module import LightningVariationalHiddenMarkov, Stage
from src.pyro_messengers.handlers import mean, force
from torch.nn.functional import mse_loss
from pyro.poutine.handlers import trace
import  matplotlib.pyplot as plt
plt.ioff()
class LightningMSEHiddenMarkov(LightningVariationalHiddenMarkov):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.automatic_optimization = True
    def configure_optimizers(self):
        return self.optimizer_cls(self.hidden_markov_model.parameters())

    def training_step(self, batch: torch.Tensor):
        guide_trace = trace(self.variational_distribution).get_trace(batch)
        with mean():
            with force(trace = guide_trace,forcing_interval=1, latent_group_name="latent", alpha=0.05,subspace_dim = 1):
                z_tensor, x_tensor = self.hidden_markov_model.get_history(batch)

        loss = mse_loss(x_tensor, batch)
        self.log("mse_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.update_metric_collection(Stage.train, batch)
        return loss






