# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Dict, Optional,overload, ParamSpec, TypeVar, Callable, Union
from src.utils.trace_utils import is_group_msg

from pyro.poutine.messenger import Messenger

import torch

from pyro.poutine.runtime import Message
from pyro.poutine.trace_struct import Trace


_P = ParamSpec("_P")
_T = TypeVar("_T")

class SubSpaceReplayMessenger(Messenger):


    def __init__(
        self,
        subspace_dim: int,
        group_name: str,
        trace: "Trace" = None
    ) -> None:
        """
        :param trace: a trace whose values should be reused

        Constructor.
        Stores trace in an attribute.
        """

        self.subspace_dim=subspace_dim
        self.group_name=group_name
        self.trace = trace
        super().__init__()

    def _postprocess_message(self, msg: Message) -> None:
        if not is_group_msg(msg,self.group_name):
            return

        guide_msg = self.trace.nodes[msg["name"]]
        guide_value = guide_msg["value"]


        msg["done"] = True
        msg["infer"] = guide_msg["infer"]
        msg["value"][..., :self.subspace_dim] = guide_value[..., :self.subspace_dim]