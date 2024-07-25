from typing import TYPE_CHECKING, Any, Literal, Optional, Union, Sequence, Dict
import torch
import datetime
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.handlers import trace
from pyro.poutine import Trace
from torchmetrics import Accuracy
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from src.utils.custom_typehint import TensorIterable
import random
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric, MetricCollection
from src.metrics.pse import  get_average_spectrum
import numpy as np
from src.metrics.klx import klx_metric
if TYPE_CHECKING:
    from src.models.hidden_markov_model import HiddenMarkovModel
from src.models.forecaster import Forecaster
from src.constants import HIDDEN_VARIABLE_NAME as _Z, OBSERVED_VARIABLE_NAME as _X
from src.training.messengers.handlers import force, observe
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
class NStepMeanSquaredError(MeanSquaredError):

    def __init__(self,n_steps: int, n_samples: int, time_dimension: int = -2, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.n_samples = n_samples

        self.time_dimension = time_dimension


    def cat_truncate(self, x: TensorIterable) -> torch.Tensor:
        return torch.cat([x_[-self.n_steps:] for x_ in x])
    def update(self, forecaster: Forecaster,x_true: TensorIterable) -> None:

        forecasts = forecaster(x_true,self.n_steps,self.n_samples, probabilistic=False, truncate_batch=True)

        predicted_values = self.cat_truncate(forecasts)
        actual_values = self.cat_truncate(x_true)

        super().update(predicted_values, actual_values)

class GaussianMaximumMeanDiscrepancy(Metric):

    def __init__(self, bandwidth: float):
        super().__init__()
        self.add_state("diff_scalar_product", default=[])
        self.bandwidth = bandwidth

    def update(self, forecaster: Forecaster, x_true: TensorIterable) -> None:


        x_gen = forecaster.model.generate_time_series_from_batch(x_true)

        expanded_tensors = self.sample_pad(x_true)

        diff_scalar_products = [self.get_scalar_diff(X, Y) for X, Y in zip(x_gen,expanded_tensors)]


        self.diff_scalar_product.extend(diff_scalar_products)

    def get_scalar_diff(self, X, Y):
        XX = (X**2).sum(1)
        YY = (Y**2).sum(1)
        XY = X @ Y.T
        diff_scalar_product = XX.reshape(-1,1) - 2 * XY.T + YY.reshape(1, -1)
        return diff_scalar_product

    def sample_pad(self, x_true):
        x_true_max_length = max(map(len, x_true))
        expanded_tensors = []
        for tensor in x_true:
            instances_to_sample = x_true_max_length - len(tensor)
            sampled_indices = random.sample(range(len(tensor)), instances_to_sample)
            sampled_x = tensor[sampled_indices]
            expanded_tensor = torch.cat([tensor, sampled_x])
            expanded_tensors.append(expanded_tensor)
        return expanded_tensors

    def kernel(self,x:torch.Tensor):
        return torch.exp(- x / (2*self.bandwidth**2) )
    def compute(self):
        v = [self.kernel(x) for x in self.diff_scalar_product]
        mmd = 0
        for x in v:
            offdiag_x = x - torch.diag(x.diag())
            m = len(offdiag_x)
            correction_factor = m/(m-1)
            mmd += offdiag_x.mean() * correction_factor

        return mmd


from src.utils.trace_utils import is_group_msg_getter, get_values_from_trace, get_log_prob_from_trace

class ForceMetric(Metric):

    def __repr__(self):
        return f"ForceMetric[{self.forcing_interval}]"
    def __init__(self, forcing_interval: int):
        super().__init__()
        self.forcing_interval = forcing_interval
        self.hidden_getter = is_group_msg_getter(_Z)
        self.obs_getter = is_group_msg_getter(_X)
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
            force(hmm, latent_group_name=_Z, trace=guide_trace, forcing_interval=self.forcing_interval,
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

    def plot(self, *_: Any, **__: Any) -> Any:
        observed_trajectory = self.observed_trajectory.numpy()
        # assuming a batchsize of 1:
        observed_trajectory = observed_trajectory[0]

        for x in np.split(observed_trajectory, 1, -1):
            plt.plot(x.reshape(-1))

        #observed_log_prob  = self.observed_log_prob.numpy()

        #plt.plot(observed_log_prob)

        #hidden_log_prob = self.hidden_log_prob.numpy()
        #plt.plot(hidden_log_prob)


class PowerSpectrumCorrelation(Metric):
    higher_is_better = True

    def __init__(self, smoothing_sigma: float, frequency_cutoff: int):
        super().__init__()
        self.smoothing_sigma = smoothing_sigma
        self.frequency_cutoff = frequency_cutoff
        self.add_state("avg_spectrum_gen", default=[])
        self.add_state("avg_spectrum_true", default=[])
        self.add_state("dim_x", default=torch.tensor(0))
        self.add_state("n_total", default=torch.tensor(0))





    def update(self, forecaster: Forecaster, x_true: TensorIterable) -> None:
        x_gen = forecaster.model.generate_time_series_from_batch(x_true)
        x_gen = [x[:len(y)] for x,y in zip(x_gen,x_true)]
        dim_x = x_gen[0].shape[-1]

        self.dim_x = torch.tensor(dim_x)

        for dim in range(dim_x):
            for x,y in zip(x_gen, x_true):

                self.n_total+=1

                spectrum_gen = get_average_spectrum(x[:, dim].unsqueeze(0))
                spectrum_true = get_average_spectrum(y[:, dim].unsqueeze(0))

                self.avg_spectrum_gen.append(spectrum_gen)
                self.avg_spectrum_true.append(spectrum_true)


    def compute(self) -> Tensor:

        pse_corrs = []
        for x_true, x_gen in zip(self.avg_spectrum_true,self.avg_spectrum_gen):

            spectrum_true = x_true[:self.frequency_cutoff]
            spectrum_gen = x_gen[:self.frequency_cutoff]
            # plot_spectrum_comparison(s_true=spectrum_true, s_gen=spectrum_gen)
            pse_corr_per_dim = torch.corrcoef(x=spectrum_gen, y=spectrum_true)[0, 1]
            pse_corrs.append(pse_corr_per_dim)

        return torch.mean(torch.tensor(pse_corrs))


class KLDivergenceObservedSpace(Metric):
    #Doesnt work
    def __init__(self, n_bins: int, smoothing_alpha):
        super().__init__()
        self.n_bins = n_bins
        self.smoothing_alpha = smoothing_alpha
        self.add_state("kldivergence", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, x_gen: Tensor, x_true: TensorIterable) -> None:
        a = [klx_metric(x_gen_.detach(), x_true.detach(), self.n_bins, self.smoothing_alpha) for x_gen_ in x_gen]
        self.kldivergence = klx_metric(x_gen, x_true, self.n_bins, self.smoothing_alpha)

    def compute(self) -> Tensor:
        return self.kldivergence

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
        a = 0
        forcing_intervals = [f.forcing_interval for f in self.values()]
        cmap = plt.get_cmap('Set1')
        #norm = plt.Normalize(min(forcing_intervals), max(forcing_intervals))
        colors = cmap(forcing_intervals)
        force_metric: ForceMetric
        for force_metric in self.values():

            force_metric.plot()
            path= Path(fr"C:\Users\roman.nefedov\PycharmProjects\PLRNN_Family_Variational_Inference\plots\{self.timestamp}")
            path.mkdir(exist_ok=True,parents=True)
            plt.savefig( path / f"{force_metric.forcing_interval}.png")
            plt.close()

class PyroTimeSeriesMetricCollection(MetricCollection):
    def __init__(self,  metrics: list[Metric],n_steps: int, truncate_batch:bool = True, n_samples: int = 1000 ):
        super().__init__(metrics)
        self.n_steps = n_steps
        self.truncate_batch = truncate_batch
        self.n_samples = n_samples


