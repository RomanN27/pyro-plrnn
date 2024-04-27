import torch
from torch import nn as nn
from src.models.elementary_components import Bias

class LinearObservationModel(nn.Module):

    def __init__(self, obs_dim, z_dim):
        super().__init__()
        # initialize the three linear.yaml transformations used in the neural network
        self.linear = nn.Linear(z_dim, obs_dim)
        self.Gamma  = nn.Parameter(torch.empty(obs_dim))
        nn.init.uniform_(self.Gamma,0,0.1**0.5)


    def forward(self, z_t):

        return self.linear(z_t), self.Gamma**2

class OrderedLogitModel(nn.Module):
    #Not working yet
    def __init__(self,obs_dim,z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, obs_dim,bias=False)
        self.bias = Bias(z_dim)

    def forward(self,z_t):
        theta = self.bias
        pass

class MultionomialLink(nn.Module):
    def __init__(self,obs_dim,z_dim,n_categories):
        super().__init__()
        self.linear = nn.Linear(z_dim, obs_dim)
        self.softmax = nn.Softmax()

    def forward(self,z_t: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.linear(z_t))

class PoissonLink(nn.Module):
    def __init__(self,obs_dim,z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, obs_dim)

    def forward(self,z_t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.linear(z_t))

class ListConcat(nn.Module):

    def __init__(self,sub_models: dict[str,nn.Module]):
        #dict since the submodels are passed via hydra. I couldn't figure out how to pass the submodels as a list via the yaml config.
        # Hence the dict
        super().__init__()
        self.models = list(sub_models.values())
    def forward(self, *args,**kwargs) -> list:
        return [model(*args, **kwargs) for model in self.models]
