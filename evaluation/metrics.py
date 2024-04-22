from typing import Any, TYPE_CHECKING
import torch
from custom_typehint import TensorIterable
import random
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric, MetricCollection
from evaluation.pse import  get_average_spectrum
import numpy as np
from evaluation.klx import klx_metric
if TYPE_CHECKING:
    from time_series_model import TimeSeriesModel
from forecaster import Forecaster

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
        XX = X @ X.T
        YY = Y @ Y.T
        XY = X @ Y.T
        diff_scalar_product = XX - 2 * XY + YY
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

        self.dim_x = dim_x

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
            pse_corr_per_dim = np.corrcoef(x=spectrum_gen, y=spectrum_true)[0, 1]
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


class PyroTimeSeriesMetricCollection(MetricCollection):
    def __init__(self,  metrics: list[Metric],n_steps: int, truncate_batch:bool = True, n_samples: int = 1000 ):
        super().__init__(metrics)
        self.n_steps = n_steps
        self.truncate_batch = truncate_batch
        self.n_samples = n_samples


