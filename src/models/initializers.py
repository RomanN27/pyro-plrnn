from src.models.transition_models.plrnns.plrnns import VanillaPLRNN, dendPLRNN
from src.models.transition_models.plrnns.plrnn_components import DendriticPhi,DendriticConnector
import torch
from src.models.model_wrappers.model_wrapper import ModelWrapper
import  torch.nn as nn
import torch
from abc import  abstractmethod
class Initializer(ModelWrapper):

    @abstractmethod
    def _initialize(self, model :nn.Module)-> None:...


    @torch.no_grad()
    def __call__(self, model: nn.Module) -> None:
        self._initialize(model)



class npRNNInitialization(Initializer):
    """
    Implementation of Talathi, S. S., & Vartak, A. (2016). Improving performance of recurrent neural network with relu nonlinearity
    (arXiv:1511.03771). arXiv. [https://doi.org/10.48550/arXiv.1511.03771](https://doi.org/10.48550/arXiv.1511.03771)
    """



    def _initialize(self, model: VanillaPLRNN | dendPLRNN):
        n_dim = model.connectivity_module.off_diag.W.size(-1)

        W = self.sample_nprnn_matrix(n_dim)

        A_diag = torch.diagonal(W)

        model.diag.A_diag = nn.Parameter(A_diag)
        model.connectivity_module.off_diag.W = nn.Parameter(W)


    def sample_nprnn_matrix(self,n_dim: int) -> torch.Tensor:
        R = torch.randn(n_dim).unsqueeze(-1)
        A = R @ R.T / n_dim
        A_I = A + torch.diag(torch.ones(n_dim))
        eigenvalues = torch.linalg.eigh(A_I).eigenvalues
        e_max = eigenvalues.max()
        W = A_I / e_max
        return W



class dendPLRNNInitialization(Initializer):
    """
    Implementation of initialiazion proposed in :
    Brenner, M., Hess, F., Mikhaeil, J. M., Bereska, L., Monfared, Z., Kuo, P.-C., & Durstewitz, D. (2022).
    _Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical Systems_ (arXiv:2207.02542). arXiv.
    [https://doi.org/10.48550/arXiv.2207.02542](https://doi.org/10.48550/arXiv.2207.02542)"""

    def __init__(self,h_bound: float = 3,bound_power = -0.5):
        self.h_bound =h_bound
        self.bound_power = bound_power
    def _initialize(self, model: dendPLRNN):

        alpha, h = self.extract_alpha_and_h(model)
        self.__initialize(alpha, h)

    @staticmethod
    def extract_alpha_and_h(model:dendPLRNN):
        connector: DendriticConnector = model.connectivity_module
        phi: DendriticPhi = connector.phi
        alpha = phi.alpha
        h = phi.H
        return alpha, h


    def __initialize(self,alpha, h):
        B = alpha.size(-1)
        bound = B ** self.bound_power
        nn.init.uniform_(alpha, -bound, bound)
        nn.init.uniform_(h, -self.h_bound, self.h_bound )


