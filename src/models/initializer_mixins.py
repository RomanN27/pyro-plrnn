import torch
import torch.nn as nn

from src.models.transition_models.plrnns.plrnn_base import PLRNN
from src.models.transition_models.plrnns.raw_plrnns import _DendPLRNN, _ShallowPLRNN


class NormalizedPositiveDefiniteInitializer:
    """
    Implementation of Talathi, S. S., & Vartak, A. (2016). Improving performance of recurrent neural network with relu nonlinearity
    (arXiv:1511.03771). arXiv. [https://doi.org/10.48550/arXiv.1511.03771](https://doi.org/10.48550/arXiv.1511.03771)
    """

    def __init__(self: PLRNN, **kwargs):
        super().__init__(**kwargs)
        self.initialize_A_and_W()

    def initialize_A_and_W(self: PLRNN):
        n_dim = self.off_diag.W.size(-1)
        W = self.sample_normalized_positive_definite_matrix(n_dim)
        A_diag = torch.diagonal(W)
        self.diag.A_diag.data = A_diag
        self.off_diag.W.data = W

    @staticmethod
    def sample_normalized_positive_definite_matrix(n_dim: int) -> torch.Tensor:
        R = torch.randn(n_dim).unsqueeze(-1)
        A = R @ R.T / n_dim
        A_I = A + torch.diag(torch.ones(n_dim))
        eigenvalues = torch.linalg.eigh(A_I).eigenvalues
        e_max = eigenvalues.max()
        W = A_I / e_max
        return W


class ZeroBiasInitializer:
    def __init__(self: PLRNN, **kwargs):
        super().__init__(**kwargs)
        self.bias.data.zero_()


class UniformAlphaInitializer:
    def __init__(self: _DendPLRNN, **kwargs):
        super().__init__(**kwargs)
        alpha = self.phi.alpha
        B = alpha.size(-1)
        self.uniform_alpha_initialize(B, alpha)

    @staticmethod
    def uniform_alpha_initialize(B, alpha):
        bound = B ** (-0.5)
        nn.init.uniform_(alpha.data, -bound, bound)


class UniformThresholdHInitializer:


    def __init__(self: _DendPLRNN, min_: float, max_: float, **kwargs):
        super().__init__(**kwargs)
        H = self.phi.H
        nn.init.uniform_(H.data,min_,max_)

class ShallowInitializer:
    def __init__(self:_ShallowPLRNN,**kwargs):
        super().__init__(**kwargs)
        self.apply(self.shallow_weights_init)

    def shallow_weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)