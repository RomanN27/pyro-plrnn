import datetime
from typing import Any, Optional, Union, Dict, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyro.poutine import trace, Trace
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from src.utils.variable_group_enum import V
from src.models.hidden_markov_model import HiddenMarkovModel
from src.training.messengers import force
from src.utils.trace_utils import is_group_msg_getter, get_values_from_trace, get_log_prob_from_trace


class ForceMetric(Metric):

    def __repr__(self):
        return f"ForceMetric[{self.forcing_interval}]"
    def __init__(self, forcing_interval: int):
        super().__init__()
        self.forcing_interval = forcing_interval
        self.hidden_getter = is_group_msg_getter(V.LATENT)
        self.obs_getter = is_group_msg_getter(V.OBSERVED)
        self.add_state("observed_trajectory", default=torch.tensor(0.))
        self.add_state("observed_log_prob", default=torch.tensor(0.))
        self.add_state("hidden_log_prob", default=torch.tensor(0.))



    def update(self, batch: torch.Tensor, hmm: "HiddenMarkovModel",guide) -> None:
        with torch.no_grad():
            guide_trace = trace(guide).get_trace(batch)

            trace_ = self.get_step_forced_trace(batch, guide_trace, hmm)

            trace_.compute_log_prob()

            x_probs, x_values, z_probs = self.get_probs_and_values(trace_)

            x_probs = torch.stack(x_probs)
            x_values = torch.stack(x_values,1)
            z_probs = torch.stack(z_probs)
            self.observed_trajectory = x_values
            self.observed_log_prob = x_probs
            self.hidden_log_prob = z_probs

    def get_step_forced_trace(self, batch, guide_trace, hmm):
        trace_ = trace(
            force(hmm, latent_group_name=V.LATENT, trace=guide_trace, forcing_interval=self.forcing_interval,
                  subspace_dim=None, alpha=0.)
        ).get_trace(batch)
        return trace_


    def get_probs_and_values(self, trace: Trace):
        x_values = get_values_from_trace(trace, self.obs_getter)
        x_probs = get_log_prob_from_trace(trace, self.obs_getter)
        z_probs = get_log_prob_from_trace(trace, self.hidden_getter)
        return x_probs, x_values, z_probs

    def compute(self) -> Any:
        pass

    def plot(self, ax = None) -> Any:
        fig, ax = (None, ax) if ax is not None else plt.subplots()
        observed_trajectory = self.observed_trajectory.numpy()
        # assuming a batchsize of 1:
        observed_trajectory = observed_trajectory[0]

        for x in np.split(observed_trajectory, 1, -1):
            ax.plot(x.reshape(-1),label = repr(self))
        ax.legend()
        return fig, ax


class ForceMetrics(MetricCollection):

    def __init__(self, forcing_intervals:list[int]):
        force_metrics = [ForceMetric(forcing_interval) for forcing_interval in forcing_intervals]
        force_metrics = {repr(force_metric):force_metric for force_metric in force_metrics}
        self.timestamp = datetime.datetime.now().strftime("%H-%M")
        super().__init__(force_metrics,compute_groups=False)


    def plot(
        self,
        val: Optional[Union[Dict, Sequence[Dict]]] = None,
        ax: Optional[Union[_AX_TYPE, Sequence[_AX_TYPE]]] = None,
        together: bool = False,
    ) -> Sequence[_PLOT_OUT_TYPE]:
        num_axes = len(self)
        num_cols = int(np.ceil(np.sqrt(num_axes)))
        num_rows = int(np.ceil(num_axes / num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,figsize= (20,10))
        for v,ax in zip(self.values(),axs.reshape(-1).tolist()):
            v.plot(ax=ax)
        return fig,axs
