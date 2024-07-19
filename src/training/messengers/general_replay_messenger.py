
from typing import TYPE_CHECKING, Dict, Optional,Callable
from src.training.messengers import get_time_stamp

from pyro.poutine.replay_messenger import ReplayMessenger
import re
from abc import  ABC, abstractmethod

from pyro.poutine.handlers import _make_handler
import re
import torch
if TYPE_CHECKING:

    from pyro.poutine.runtime import Message
    from pyro.poutine.trace_struct import Trace


class GeneralTraceReplayMessenger(ReplayMessenger):
    def __init__(self, trace: "Trace", msg_filter:Optional["MsgFilter"] = None,
                 updater: Optional["StandardUpdate"] = None,
                 post_process_updater:Optional["StandardUpdate"] = None):
        super().__init__(trace=trace)
        self.msg_filter = msg_filter if msg_filter is not None else lambda x: True
        self.update_fun = updater
        self.post_process_updater =post_process_updater

    def get_corresponding_trace_message(self, msg: "Message")-> "Message":
        return self.trace.nodes[msg["name"]]



    def _pyro_sample(self, msg: "Message") -> None:
        if not self.msg_filter(msg):
            return

        if self.update_fun is None:
            super()._pyro_sample(msg)
            return

        self.update_fun(msg,self.get_corresponding_trace_message(msg))

    def _postprocess_message(self, msg: "Message") -> None:
        if not self.msg_filter(msg):
            return
        if self.post_process_updater is not None:
            self.post_process_updater(msg,self.get_corresponding_trace_message(msg))


class MsgFilter(ABC):
    @abstractmethod
    def __call__(self, msg: "Message")->bool:...

class IntervalFilter(MsgFilter):

    def __init__(self, interval: int):
        self.interval = interval

    def __call__(self, msg: "Message")-> bool:
        name = msg["name"]
        time_stamp = get_time_stamp(name)
        return not (time_stamp % self.interval)

class GroupNameFilter(MsgFilter):
    def __init__(self, group_name: str):
        self.group_name = group_name


    def __call__(self, msg: "Message")-> bool:
        name = msg["name"]
        boo = bool(re.match(f"{self.group_name}_\d*$", name))
        return boo

class ChainFilter(MsgFilter):
    def __init__(self, *filters: MsgFilter):
        self.filters= filters

    def __call__(self,msg: "Message"):
        boo = True
        for filter_ in self.filters:
            boo&=filter_(msg)
            if not boo:
                return False
        return True





class StandardUpdate:

    def update(self, msg: "Message", trace_msg: "Message"):
        _, msg["value"] = self(msg["value"],trace_msg["value"])
        msg["done"] = True
        msg["infer"] = trace_msg["infer"]

    def __call__(self, msg_value: torch.Tensor, trace_msg_value: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        return msg_value, trace_msg_value



class AlphaUpdater(StandardUpdate):
    def __init__(self,alpha):
        self.alpha = alpha

    def __call__(self,msg_value: torch.Tensor, trace_msg_value: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        return msg_value, msg_value * self.alpha + trace_msg_value * (1-self.alpha)

class SubspaceUpdater(StandardUpdate):
    def __init__(self, sub_space_dim: int):
        self.sub_space_dim = sub_space_dim

    def __call__(self, msg_value: torch.Tensor, trace_msg_value: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        return msg_value[..., :self.sub_space_dim], trace_msg_value[..., :self.sub_space_dim]

class ChainUpdater(StandardUpdate):

    def __init__(self,*updaters: StandardUpdate):
        self.updaters = updaters

    def __call__(self, msg_value: torch.Tensor, trace_msg_value: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        current_msg_value = msg_value
        current_trace_msg_value =trace_msg_value

        for updater in self.updaters:
            current_msg_value, current_trace_msg_value = updater(current_msg_value, current_trace_msg_value)

        return current_msg_value, current_trace_msg_value