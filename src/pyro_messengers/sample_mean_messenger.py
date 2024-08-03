# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pyro.distributions import Delta
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import Message
from src.utils.variable_group_enum import V

class SampleMeanMessenger(Messenger):

    def _process_message(self, msg: Message) -> None:

        if V.is_variable_group(msg["name"]):
            mean = msg["fn"].mean
            msg["fn"] = Delta(mean)
