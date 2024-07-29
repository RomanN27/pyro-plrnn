from src.models.transition_models.plrnns.plrnn_base import PLRNN
from src.models.transition_models.plrnns.plrnn_components import VanillaConnector, DendriticConnector, ShallowConnector, ClippedDendriticConnector, ClippedDendriticPhi


class VanillaPLRNN(PLRNN):

    def __init__(self, z_dim: int):
        vanilla_connector = VanillaConnector(z_dim)
        super().__init__(z_dim, vanilla_connector)


class dendPLRNN(PLRNN):

    def __init__(self, z_dim: int, B: int):
        dendritic_connector = DendriticConnector(z_dim, B)
        super().__init__(z_dim, dendritic_connector)


class ClippedDendPLRNN(PLRNN):
    def __init__(self, z_dim: int, B: int):
        clipped_dendritic_connector = ClippedDendriticConnector(z_dim, B)
        super().__init__(z_dim, clipped_dendritic_connector)


class shallowPLRNN(PLRNN):
    def __init__(self, z_dim: int, hidden_dim: int):
        shallow_connector = ShallowConnector(z_dim, hidden_dim)
        super().__init__(z_dim, shallow_connector)


