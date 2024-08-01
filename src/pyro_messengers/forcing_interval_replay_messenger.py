from typing import Optional

import numpy as np
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import Message
from pyro.poutine.trace_struct import Trace

from src.utils.trace_utils import is_group_msg, get_time_stamp


#TODO implement optimal forcing interval
class ForcingIntervalReplayMessenger(Messenger):

    def __init__(
            self,
            trace: "Trace",
            forcing_interval: int,
            latent_group_name: str,
            alpha: float = 1.,
            subspace_dim: Optional[int] = None

    ) -> None:
        super().__init__()

        self.trace = trace
        self.forcing_interval = forcing_interval
        self.latent_group_name = latent_group_name
        self.alpha = alpha
        self.subspace_dim = subspace_dim
        #self.anchor_time_step = np.random.choice(range(self.forcing_interval))


    def _postprocess_message(self, msg: Message) -> None:
        if not is_group_msg(msg, self.latent_group_name):
            return

        t = get_time_stamp(msg)
        force_boo = (t-1) % self.forcing_interval

        if force_boo:
            return

        guide_msg = self.trace.nodes[msg["name"]]

        msg["done"] = True
        msg["infer"] = guide_msg["infer"]



        updated_value = msg["value"] * self.alpha + (1 - self.alpha) * guide_msg["value"].detach().clone()
        if t == 1:
            #first step is completely infered
            msg["value"] = updated_value

        msg["value"][...,:self.subspace_dim] = updated_value[...,:self.subspace_dim]
