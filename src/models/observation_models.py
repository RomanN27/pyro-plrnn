import torch
from torch import nn as nn



class LinearObservationModel(nn.Module):

    def __init__(self, obs_dim, z_dim):
        super().__init__()
        # initialize the three linear.yaml transformations used in the neural network
        self.linear = nn.Linear(z_dim, obs_dim)
        self.Gamma  = nn.Parameter(torch.empty(obs_dim))
        nn.init.uniform_(self.Gamma,0,0.1**0.5)


    def forward(self, z_t):

        return self.linear(z_t), self.Gamma**2

class IdentityObservationModel(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        # initialize the three linear.yaml transformations used in the neural network
        self.obs_dim = obs_dim
        self.Gamma = nn.Parameter(torch.empty(obs_dim))
        nn.init.uniform_(self.Gamma, 0, 0.1 ** 0.5)

    def forward(self, z_t):
        return z_t[...,:self.obs_dim], torch.ones_like(self.Gamma) * 0.01
class OrderedLogitModel(nn.Module):
    #Not working yet
    def __init__(self,obs_dim,z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, obs_dim,bias=False)
        self.bias = torch.ones(z_dim)

    def forward(self,z_t):
        theta = self.bias
        pass

class MultionomialLink(nn.Module):
    def __init__(self,obs_dim,z_dim,n_categories):
        super().__init__()
        self.linear = nn.Linear(z_dim, obs_dim*n_categories)
        self.softmax = nn.Softmax(dim=-1)
        self.n_categories = n_categories


    def forward(self,z_t: torch.Tensor) -> tuple[torch.Tensor]:
        shape = z_t.shape
        x = self.linear(z_t)
        x = x.view(shape[:-1] + (-1,self.n_categories))
        return self.softmax(x),

class PoissonLink(nn.Module):
    def __init__(self,obs_dim,z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, obs_dim)

    def forward(self,z_t: torch.Tensor) -> tuple[torch.Tensor]:
        return torch.exp(self.linear(z_t)),

class ListConcat(nn.Module):

    def __init__(self,sub_models: dict[str,nn.Module]):
        #dict since the submodels are passed via hydra. I couldn'delta_t figure out how to pass the submodels as a list via the yaml config.
        # Hence the dict
        super().__init__()
        self.models = nn.ModuleList(sub_models.values())
    def forward(self, *args,**kwargs) -> list:
        return [model(*args, **kwargs) for model in self.models]
