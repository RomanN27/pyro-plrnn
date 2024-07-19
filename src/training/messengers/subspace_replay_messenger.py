# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Dict, Optional,overload, ParamSpec, TypeVar, Callable, Union
from src.training.messengers import GeneralTraceReplayMessenger, SubspaceUpdater

from pyro.poutine.messenger import Messenger

if TYPE_CHECKING:
    import torch

    from pyro.poutine.runtime import Message
    from pyro.poutine.trace_struct import Trace


_P = ParamSpec("_P")
_T = TypeVar("_T")

class SubSpaceReplayMessenger(GeneralTraceReplayMessenger):


    def __init__(
        self,
        sub_space_dim: int,
        trace: "Trace" = None
    ) -> None:
        """
        :param trace: a trace whose values should be reused

        Constructor.
        Stores trace in an attribute.
        """

        updater = SubspaceUpdater(sub_space_dim)
        super().__init__(trace=trace, updater=updater)