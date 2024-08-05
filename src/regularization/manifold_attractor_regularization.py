from src.models.transition_models.plrnns.plrnn_base import PLRNN
import torch
class ManifoldAttractorRegularization:

    def __init__(self, lambda_: float, n_of_target_points: int):
        self.lambda_ = lambda_
        self.n_of_regularized_latent_states = n_of_target_points

    def __call__(self,plrnn_model: PLRNN):
        diag = plrnn_model.diag.A_diag[...,:self.n_of_regularized_latent_states]
        diag_loss = torch.sum((diag - 1) ** 2)
        off_diag = plrnn_model.off_diag.W[:self.n_of_regularized_latent_states]
        off_diag_loss = torch.sum(off_diag ** 2)
        bias = plrnn_model.bias[:self.n_of_regularized_latent_states]
        bias_loss = torch.sum(bias ** 2)

        reg_loss = diag_loss + off_diag_loss + bias_loss
        reg_loss = self.lambda_ * reg_loss

        return reg_loss
