from typing import TYPE_CHECKING
import torch

from src.utils.custom_typehint import TensorIterable
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection
from src.metrics.metric_base import Metric
from src.metrics.pse import get_average_spectrum
from src.metrics.klx import klx_metric

if TYPE_CHECKING:
    pass
from src.models.forecaster import Forecaster


class NStepMeanSquaredError(MeanSquaredError):

    def __init__(self, n_steps: int, n_samples: int, time_dimension: int = -2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.n_samples = n_samples

        self.time_dimension = time_dimension

    def cat_truncate(self, x: TensorIterable) -> torch.Tensor:
        return torch.cat([x_[-self.n_steps:] for x_ in x])

    def update(self, forecaster: Forecaster, x_true: TensorIterable) -> None:
        forecasts = forecaster(x_true, self.n_steps, self.n_samples, probabilistic=False, truncate_batch=True)

        predicted_values = self.cat_truncate(forecasts)
        actual_values = self.cat_truncate(x_true)

        super().update(predicted_values, actual_values)


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

    def update(self, forecaster: Forecaster, batch: TensorIterable) -> None:
        x_gen = forecaster.model.generate_time_series_from_batch(batch)
        x_gen = [x[:len(y)] for x, y in zip(x_gen, batch)]
        dim_x = x_gen[0].shape[-1]

        self.dim_x = torch.tensor(dim_x)

        for dim in range(dim_x):
            for x, y in zip(x_gen, batch):
                self.n_total += 1

                spectrum_gen = get_average_spectrum(x[:, dim].unsqueeze(0))
                spectrum_true = get_average_spectrum(y[:, dim].unsqueeze(0))

                self.avg_spectrum_gen.append(spectrum_gen)
                self.avg_spectrum_true.append(spectrum_true)

    def compute(self) -> Tensor:

        pse_corrs = []
        for x_true, x_gen in zip(self.avg_spectrum_true, self.avg_spectrum_gen):
            spectrum_true = x_true[:self.frequency_cutoff]
            spectrum_gen = x_gen[:self.frequency_cutoff]
            # plot_spectrum_comparison(s_true=spectrum_true, s_gen=spectrum_gen)
            pse_corr_per_dim = torch.corrcoef(x=spectrum_gen, y=spectrum_true)[0, 1]
            pse_corrs.append(pse_corr_per_dim)

        return torch.mean(torch.tensor(pse_corrs))




class PyroTimeSeriesMetricCollection(MetricCollection):
    def __init__(self, metrics: list[Metric], n_steps: int, truncate_batch: bool = True, n_samples: int = 1000):
        super().__init__(metrics)
        self.n_steps = n_steps
        self.truncate_batch = truncate_batch
        self.n_samples = n_samples
