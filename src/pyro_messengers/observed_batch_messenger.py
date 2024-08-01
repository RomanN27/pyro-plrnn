# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, ParamSpec, TypeVar

import torch
from pyro.poutine.messenger import Messenger
from src.pyro_messengers import GroupNameFilter

if TYPE_CHECKING:
    from pyro.poutine.runtime import Message

from src.utils.trace_utils import get_time_stamp

_P = ParamSpec("_P")
_T = TypeVar("_T")


class ObservedBatchMessenger(Messenger):
    def __init__(self, batch: torch.tensor, observation_group_symbol: str) -> None:
        super().__init__()
        self.batch = batch
        self.filter = GroupNameFilter(observation_group_symbol)

    def _pyro_sample(self, msg: "Message") -> None:
        if not self.filter(msg):
            return

        name = msg["name"]
        t = get_time_stamp(name)

        msg["value"] = self.batch[:, t - 1, :]
        msg["is_observed"] = True
