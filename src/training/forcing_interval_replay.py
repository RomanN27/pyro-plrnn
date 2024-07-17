
from typing import TYPE_CHECKING, Dict, Optional

from pyro.poutine.messenger import Messenger
import re

from pyro.poutine.handlers import _make_handler

if TYPE_CHECKING:
    import torch
    from pyro.poutine.runtime import Message
    from pyro.poutine.trace_struct import Trace
#TODO implement optimal forcing interval
class ForcingIntervalReplayMessenger(Messenger):
    HIDDEN_VARIABLE_NAME = "z"
    def __init__(
        self,
        trace: "Trace" ,
            forcing_interval: int
    ) -> None:

        super().__init__()
        self.forcing_interval = forcing_interval
        self._n_sampled_time_steps = 0
        self.trace = trace


    def get_time_stamp(self, site_name: str):
        pattern = fr"^{self.HIDDEN_VARIABLE_NAME}_(\d*)"
        time_stamp = re.search(pattern, site_name)
        time_stamp = int(time_stamp.groups()[0]) if time_stamp else time_stamp
        return time_stamp


    def _pyro_sample(self, msg: "Message") -> None:
        name = msg["name"]
        time_stamp = self.get_time_stamp(name)
        if time_stamp is not None:
            if not (time_stamp % self.forcing_interval):
                guide_msg = self.trace.nodes[name]
                msg["done"] = True
                msg["value"] = guide_msg["value"]
                msg["infer"] = guide_msg["infer"]


@_make_handler(ForcingIntervalReplayMessenger)
def force(fn: "Trace", forcing_interval: int):
    ...
