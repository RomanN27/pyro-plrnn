from typing import Type, Optional, TypeVar, Generic
from uuid import uuid4

import pyro
import torch
from lightning import LightningModule
from pyro.distributions import Distribution
from torch import nn as nn

random_name = lambda: str(uuid4())

T = TypeVar("T", bound=torch.Tensor | None)
ModelType = TypeVar("ModelType", bound=nn.Module )
DistType = TypeVar("DistType", bound=Distribution)


class ModelBasedSampler(Generic[T, ModelType, DistType], nn.Module):
    def __init__(self, model: ModelType, distribution: DistType):
        super().__init__()
        self.model = model
        self.distribution = distribution

    def forward(self, input_tensor: T, name: str = random_name()) -> torch.Tensor:
        dist_parameters = self.model(input_tensor)
        dist = self.distribution(
            *dist_parameters)  #typecheck error because Distribuiton itself does not implement init however all subclasses do
        return pyro.sample(name, dist)







