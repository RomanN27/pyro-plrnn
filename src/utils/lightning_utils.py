from typing import TYPE_CHECKING

from torch import nn as nn

if TYPE_CHECKING:
    pass


def update_state_dict(lightning_module: nn.Module, state_dict_to_use: dict):
    initial_state = lightning_module.state_dict()
    initial_state.update(state_dict_to_use)
    lightning_module.load_state_dict(initial_state)


def update_mean(old_mean, new_value, n):
    p  = 1/(n+1)
    new_mean = (1-p) * old_mean + p * new_value
    return new_mean
