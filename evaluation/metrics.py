from typing import Any, TYPE_CHECKING
import torch
from custom_typehint import TensorIterable
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric, MetricCollection
from pse import  get_average_spectrum
import numpy as np
from klx import klx_metric
if TYPE_CHECKING:
    from time_series_model import TimeSeriesModel
from forecaster import Forecaster

class NStepMeanSquaredError(MeanSquaredError):

    def __init__(self,n_steps: int,time_dimension: int = -2, *args,**kwargs):
        self.n_steps = n_steps
        self.time_dimension = time_dimension
        super().__init__(*args,**kwargs)
    def update(self, x_gen: Tensor, x_true: Tensor) -> None:
        target_length = self.n_steps
        t_start = x_gen.size(self.time_dimension) - target_length
        x_gen = x_gen.narrow(self.time_dimension, t_start, target_length)
        target = x_true.narrow(self.time_dimension, t_start, target_length)
        super().update(x_gen, target)


class PowerSpectrumCorrelation(Metric):
    higher_is_better = True

    def __init__(self, smoothing_sigma: float, frequency_cutoff: int):
        self.smoothing_sigma = smoothing_sigma
        self.frequency_cutoff = frequency_cutoff
        self.add_state("avg_spectrum_gen", default=[])
        self.add_state("avg_spectrum_true", default=[])
        super().__init__()

    def update(self, x_gen: Tensor, x_true: Tensor) -> None:
        assert x_true.shape[1] == x_gen.shape[1]
        assert x_true.shape[2] == x_gen.shape[2]
        self.dim_x = dim_x= x_gen.shape[2]
        for dim in range(dim_x):
            spectrum_true = get_average_spectrum(x_true[:, :, dim])
            spectrum_gen = get_average_spectrum(x_gen[:, :, dim])
            if len(self.avg_spectrum_gen)  < dim_x:
                self.avg_spectrum_gen.append(spectrum_gen)
                self.avg_spectrum_true.append(spectrum_true)
            else:
                #TODO Implement proper update rule
                self.avg_spectrum_gen[dim] = spectrum_gen
                self.avg_spectrum_true[dim] = spectrum_true

    def compute(self) -> Tensor:
        pse_corrs_per_dim = []
        for dim in range(self.dim_x):
            spectrum_true = self.avg_spectrum_true[dim][:self.frequency_cutoff]
            spectrum_gen = self.avg_spectrum_gen[dim][:self.frequency_cutoff]
            # plot_spectrum_comparison(s_true=spectrum_true, s_gen=spectrum_gen)
            pse_corr_per_dim = np.corrcoef(x=spectrum_gen, y=spectrum_true)[0, 1]
            pse_corrs_per_dim.append(pse_corr_per_dim)

        return np.array(pse_corrs_per_dim).mean(axis=0)


class KLDivergenceObservedSpace(Metric):
    def __init__(self, n_bins: int, smoothing_alpha):
        self.n_bins = n_bins
        self.smoothing_alpha = smoothing_alpha
        self.add_state("kldivergence", default=torch.tensor(0), dist_reduce_fx="sum")
        super().__init__()

    def update(self, x_gen: Tensor, x_true: Tensor) -> None:
        self.kldivergence = klx_metric(x_gen, x_true, self.n_bins, self.smoothing_alpha)

    def compute(self) -> Tensor:
        return self.kldivergence


class PyroTimeSeriesMetricCollection(MetricCollection):
    def __init__(self, n_steps: int, truncate_batch:bool = True, n_samples: int = 1000, *args, **kwargs):
        self.n_steps = n_steps
        self.truncate_batch = truncate_batch
        self.n_samples = n_samples
        super().__init__(*args, **kwargs)
    def update_from_model(self,forecaster: Forecaster, x_true: TensorIterable):

        x_gen = forecaster(x_true,self.n_steps,self.n_samples,truncate_batch=self.truncate_batch)
        batch_dim = 1
        for x_gen_batch,x_true_batch in zip(x_gen.transpose(0,batch_dim), x_true):
            self.update(x_gen_batch,x_true_batch)


