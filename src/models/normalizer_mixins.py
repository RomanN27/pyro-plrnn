import torch
from src.models.transition_models.plrnns.plrnn_base import PLRNN


class PLRNNNormalizerMixin:

    def __init__(self: PLRNN, **kwargs):
        super().__init__(**kwargs)

        self.forward = self.mean_centered_plrnn_forward

    def mean_centered_plrnn_forward(self: PLRNN, z: torch.Tensor):
        return self.diag(z) + self.off_diag(self.phi(z - z.mean(-1, keepdim=True))) + self.bias
