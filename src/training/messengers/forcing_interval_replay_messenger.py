
from typing import TYPE_CHECKING, Dict, Optional
from src.training.messengers import (get_time_stamp, GeneralTraceReplayMessenger, AlphaUpdater, StandardUpdate,
                                     ChainUpdater, ChainFilter, GroupNameFilter,IntervalFilter, SubspaceUpdater)

from pyro.poutine.messenger import Messenger
import re

from pyro.poutine.handlers import _make_handler

if TYPE_CHECKING:
    import torch
    from pyro.poutine.runtime import Message
    from pyro.poutine.trace_struct import Trace
#TODO implement optimal forcing interval
class ForcingIntervalReplayMessenger(GeneralTraceReplayMessenger):
    def __init__(
        self,
        trace: "Trace" ,
            forcing_interval: int,
            latent_group_name: str,
            alpha: float = 1.,
            subspace_dim: Optional[int] = None

    ) -> None:
        msg_filter = ChainFilter(IntervalFilter(forcing_interval),GroupNameFilter(latent_group_name))
        post_process_updater = ChainUpdater(AlphaUpdater(alpha), SubspaceUpdater(subspace_dim))

        super().__init__(trace=trace, msg_filter =msg_filter, post_process_updater =post_process_updater )

