# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Dict, Union,ParamSpec, TypeVar, Callable

import pyro
import torch

from pyro.poutine.messenger import Messenger
from pyro.poutine.trace_struct import Trace
from pyro.poutine.handlers import _make_handler
if TYPE_CHECKING:
    from pyro.poutine.runtime import Message

from pyro.distributions import Normal

_P = ParamSpec("_P")
_T = TypeVar("_T")

class ObservedBatchMessenger(Messenger):
    def __init__(self, batch: torch.tensor) -> None:

        super().__init__()
        self.batch = batch

    def _pyro_sample(self, msg: "Message") -> None:

        name = msg["name"]
        t = int(name.split("_")[1])
        msg["value"] = self.batch[:,t-1,:]
        msg["is_observed"] = True


@_make_handler(ObservedBatchMessenger)
def observe(fn:Callable[_P, _T],batch:torch.Tensor)-> Callable[_P, _T]:...



def model():
    return pyro.sample("x_1",Normal(torch.zeros((100,12)),1))

batch = torch.randn((100,12,12))

m = observe(model, batch)

m()