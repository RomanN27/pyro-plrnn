import datetime
from typing import Any, Optional, Union, Dict, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyro.poutine import trace, Trace
from src.metrics.metric_base import Metric, MetricCollection, MetricLogType, Logger
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from src.utils.variable_group_enum import V
from src.models.hidden_markov_model import HiddenMarkovModel
from src.pyro_messengers import force, mean
from src.utils.trace_utils import is_group_msg_getter, get_values_from_trace, get_log_prob_from_trace
import plotly.graph_objects as go

class ForceMetric(Metric):
    log_types = [MetricLogType.plotly_figure]


    def __repr__(self):
        return f"ForceMetric[{self.forcing_interval},{str(self.alpha).replace('.','_')}]"
    def __init__(self, forcing_interval: int,subspace_dim:int = None, alpha: float = 0.):
        super().__init__()
        self.forcing_interval = forcing_interval
        self.subspace_dim = subspace_dim
        self.alpha = alpha
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
        trace_ =trace(mean(
            force(hmm, latent_group_name=V.LATENT, trace=guide_trace, forcing_interval=self.forcing_interval,
                  subspace_dim=self.subspace_dim, alpha=self.alpha)
        )).get_trace(batch)
        return trace_


    def get_probs_and_values(self, trace: Trace):
        x_values = get_values_from_trace(trace, self.obs_getter)
        x_probs = get_log_prob_from_trace(trace, self.obs_getter)
        z_probs = get_log_prob_from_trace(trace, self.hidden_getter)
        return x_probs, x_values, z_probs

    def compute(self) -> Any:
        pass

    def plot(self, ax=None) -> Any:
        fig = go.Figure()
        observed_trajectory = self.observed_trajectory.numpy()
        # assuming a batchsize of 1:
        observed_trajectory = observed_trajectory[0, ..., 0]

        for x in np.split(observed_trajectory, 1, -1):
            fig.add_trace(go.Scatter(y=x.reshape(-1), mode='lines', name=repr(self)))

        fig.update_layout(legend=dict(title="Legend"))
        return fig


class ForceMetrics(MetricCollection):

    def __init__(self, forcing_intervals:list[int],alphas: None, subspace_dim:int = None):
        alphas = [0. for _ in forcing_intervals] if alphas is None else alphas
        force_metrics = [ForceMetric(forcing_interval,subspace_dim,alpha ) for forcing_interval,alpha in zip(forcing_intervals,alphas)]
        force_metrics = {repr(force_metric):force_metric for force_metric in force_metrics}
        self.timestamp = datetime.datetime.now().strftime("%H-%M")
        self.n = 0
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
