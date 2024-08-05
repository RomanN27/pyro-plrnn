from src.models.hidden_markov_model import HiddenMarkovModel
from pyro.poutine import trace
from pyro import plate
from typing import Callable
import torch
from typing import Any

from src.utils.hydra_utils import get_module_from_cfg_path, get_module_from_relative_cfg_path
from src.utils.trace_utils import get_values_from_trace,is_group_msg_getter
from src.utils.variable_group_enum import V
from src.pyro_messengers.handlers import mean

from contextlib import nullcontext


def get_forecast_tensors_from_trace(tracer):
    observation_forecasts = get_values_from_trace(tracer.trace, is_group_msg_getter(V.OBSERVED))
    latent_forecasts = get_values_from_trace(tracer.trace, is_group_msg_getter(V.LATENT))
    observation_forecast_tensor = torch.stack(observation_forecasts, -2)
    latent_forecast_tensor = torch.stack(latent_forecasts, -2)
    return latent_forecast_tensor, observation_forecast_tensor


def get_time_range(batch, n_timesteps_to_forecast):
    t_prediction_start = batch.size(-2) + 1
    t_prediction_end = t_prediction_start + n_timesteps_to_forecast
    time_range = range(t_prediction_start, t_prediction_end)
    return time_range


class Forecaster:
    PRED_PREFIX = "pred"

    def __init__(self, model: HiddenMarkovModel, guide: Callable[[torch.Tensor], Any]) -> None:
        self.model = model
        self.guide = guide

    def __call__(self, batch: torch.Tensor, n_timesteps_to_forecast: int, n_samples: int, probabilistic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:

        with mean() if not probabilistic else plate("_num_posterior_samples", n_samples,dim=-2):
                Z = self.guide(batch)

                last_z = Z[-1]
                z_sample  =self.model.initial_sampler(None, name=f"{V.LATENT}_1")
                z_sample = z_sample.expand(n_samples, *z_sample.size()).clone()
                z_sample[...,:last_z.size(-1)] = last_z

        time_range = get_time_range(batch, n_timesteps_to_forecast)

        with trace() as tracer :
            with mean() if not probabilistic else nullcontext():
                self.model.run_over_time_range(z_sample,time_range)

        latent_forecast_tensor, observation_forecast_tensor = get_forecast_tensors_from_trace(tracer)

        return observation_forecast_tensor,latent_forecast_tensor


if __name__ == '__main__':
    from src.models.forecaster import Forecaster
    from pathlib import Path

    module = get_module_from_relative_cfg_path("flattened_configs/flattened_config_3")

    forecaster: Forecaster = module.lightning_module.forecaster
    module.data_module.setup("fit")

    data = next(iter(module.data_module.train_dataloader()))[:1]

    observation_forecast_tensor,latent_forecast_tensor = forecaster(data, 100, 100, probabilistic=False)
