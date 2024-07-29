import torch

from src.models.model_wrappers.model_wrapper import ModelWrapper
from src.models.transition_models.plrnns.plrnn_base import PLRNN


class PLRNNMeanNormalizer(ModelWrapper):

    def __call__(self, model: PLRNN):
        def mean_centered_plrnn_forward(z: torch.Tensor):
            return model.diag(z) + model.connectivity_module(z - z.mean(-1, keepdim=True)) + model.bias

        return mean_centered_plrnn_forward
