import torch

from src.models.transition_models.plrnns.plrnn_base import PLRNN
from src.models.transition_models.plrnns.plrnn_components import  ShallowPhi, ClippedDendriticPhi, DendriticPhi, ClippedShallowPhi


class _VanillaPLRNN(PLRNN):

    def __init__(self, z_dim: int):

        super().__init__(z_dim, phi=torch.relu)


class _DendPLRNN(PLRNN[DendriticPhi]):

    def __init__(self, z_dim: int, B: int):
        dendritic_connector = DendriticPhi(z_dim, B)
        super().__init__(z_dim, phi = dendritic_connector)


class _ClippedDendPLRNN(PLRNN[ClippedDendriticPhi]):
    def __init__(self, z_dim: int, B: int):
        super().__init__(z_dim, phi=ClippedDendriticPhi(z_dim, B))


class _ShallowPLRNN(PLRNN[ShallowPhi]):
    def __init__(self, z_dim: int, hidden_dim: int):
        shallow_connector = ShallowPhi(z_dim, hidden_dim)
        super().__init__(z_dim, phi = shallow_connector)


class _ClippedShallowPLRNN(PLRNN[ClippedShallowPhi]):
    def __init__(self, z_dim: int, hidden_dim: int):
        shallow_connector = ClippedShallowPhi(z_dim, hidden_dim)
        super().__init__(z_dim, phi = shallow_connector)