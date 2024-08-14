from src.models.transition_models.plrnns.plrnn_base import PLRNN

import torch
import torch.nn as nn

class ExternalInputMixin:

    def __init__(self: PLRNN,ext_input_dim: int,z_dim: int, *args,**kwargs):
        super().__init__(z_dim,*args,**kwargs)
        self.linear = nn.Linear(ext_input_dim,z_dim)
        old_forward = self.forward
        def new_forward(z,ext_input):
            return old_forward(z) + self.linear(ext_input)

        self.forward = new_forward







