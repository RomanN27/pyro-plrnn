import torch
from src.models.transition_models.plrnns.plrnn_base import PLRNN
class PLRNNNormalizerMixin(PLRNN):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.forward = self.mean_centered_plrnn_forward

    def mean_centered_plrnn_forward(self,z: torch.Tensor):
        return self.diag(z) + self.connectivity_module(z - z.mean(-1, keepdim=True)) + self.bias