from typing import TYPE_CHECKING

from torch import nn as nn

if TYPE_CHECKING:
    from src.lightning_module import LightningVariationalHiddenMarkov


def update_state_dict(lightning_module: nn.Module, state_dict_to_use: dict):
    initial_state = lightning_module.state_dict()
    initial_state.update(state_dict_to_use)
    lightning_module.load_state_dict(initial_state)
