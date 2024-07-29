from typing import Type
from uuid import uuid4

import pyro
import torch
from lightning import LightningModule
from pyro.distributions import Distribution
from torch import nn as nn


random_name = lambda : str(uuid4())


class ModelBasedSampler(nn.Module):
    def __init__(self, model: nn.Module, distribution:Type["Distribution"]):
        super().__init__()
        self.model = model
        self.distribution =  distribution

    def forward(self, input_tensor: torch.Tensor, name:str = random_name()) -> torch.Tensor:

        dist_parameters = self.model(input_tensor)
        dist = self.distribution(*dist_parameters) #typecheck error because Distribuiton itself does not implement init however all subclasses do
        return pyro.sample(name,dist)
