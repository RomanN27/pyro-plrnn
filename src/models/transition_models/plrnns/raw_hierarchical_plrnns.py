import torch
import torch.nn as nn
from src.models.transition_models.plrnns.plrnn_base import PLRNN
from src.models.transition_models.plrnns.plrnn_components import  ShallowPhi, ClippedDendriticPhi, DendriticPhi, ClippedShallowPhi
from src.models.model_sampler import ModelBasedSampler
from pyro import module
class _HierarchicalClippedShallowPLRNN(nn.Module):
    def __init__(self,):

