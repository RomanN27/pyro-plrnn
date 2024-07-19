# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Union

import torch


from pyro.poutine.scale_messenger import ScaleMessenger
from pyro.poutine.util import is_validation_enabled
from src.training.messengers import GroupNameFilter

if TYPE_CHECKING:
    from pyro.poutine.runtime import Message


class AnnealingScaleMessenger(ScaleMessenger):

    def __init__(self,beginning_annealing_factor: int, annealing_epochs:int, latent_group_name: str):
        self.current_annealing_factor = beginning_annealing_factor
        self.delta = (1 - beginning_annealing_factor) / annealing_epochs
        self.filter = GroupNameFilter(latent_group_name)
        super().__init__(scale=self.current_annealing_factor)

    def increase_annealing_factor(self):
        self.current_annealing_factor += self.delta
        self.current_annealing_factor = max(min(1, self.current_annealing_factor), 0)

    def _process_message(self, msg: "Message"):
        if self.filter(msg):
            super()._process_message(msg)

    def on_train_epoch_end(self):
        self.increase_annealing_factor()