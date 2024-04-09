from pyro.poutine.scale_messenger import ScaleMessenger
from typing import Callable, TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from pyro.poutine.runtime import Message
class SelectiveScaleMessenger(ScaleMessenger):

    def __init__(self, scale: float | torch.Tensor, select_f: Callable[["Message"], bool]) -> object:
        self.select_f = select_f
        super().__init__(scale)

    def _process_message(self, msg: "Message") -> None:

        if self.select_f(msg):
            super()._process_message(msg)