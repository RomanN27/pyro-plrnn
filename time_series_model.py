import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from typing import Type, Callable, Tuple
from custom_typehint import TorchDistributionMixinDist
from utils import collate_fn_2
from pyro.poutine import trace, uncondition

class TimeSeriesModel(nn.Module):
    HIDDEN_VARIABLE_NAME = "z"
    OBSERVED_VARIABLE_NAME = "x"

    def __init__(self, transition_model: nn.Module,
                 observation_model: nn.Module,
                 observation_distribution: Type[TorchDistributionMixinDist],
                 transition_distribution: Type[TorchDistributionMixinDist],
                 collate_fn: Callable[[list[torch.Tensor]], tuple[torch.Tensor, torch.Tensor]] = collate_fn_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_model = transition_model
        self.observation_model = observation_model

        self.transition_distribution = transition_distribution
        self.observation_distribution = observation_distribution
        self.collate_fn = collate_fn
        self.z_0 = nn.Parameter(torch.zeros(self.transition_model.z_dim))

    def sample_observation(self, batch, batch_mask, t, *distribution_parameters):
        obs_t = pyro.sample(f"{self.OBSERVED_VARIABLE_NAME}_{t}",
                            self.observation_distribution(*distribution_parameters).mask(
                                batch_mask[:, t - 1: t]).to_event(1), obs=batch[:, t - 1, :], )
        return obs_t

    def sample_next_hidden_state(self, batch_mask, t, *distribution_parameters) -> torch.Tensor:
        z_t = pyro.sample(
            f"{self.HIDDEN_VARIABLE_NAME}_{t}",
            self.transition_distribution(*distribution_parameters)
            .mask(batch_mask[:, t - 1: t])
            .to_event(1),
        )
        return z_t

    def __call__(self, batch: list[torch.Tensor]):
        pyro.module("dmm", self)

        padded_sequence, batch_mask = self.collate_fn(batch)
        T_max = padded_sequence.size(1)
        n_batches = padded_sequence.size(0)

        z_prev = self.z_0.repeat(n_batches, 1)

        with pyro.plate("z_minibatch", n_batches):
            for t in pyro.markov(range(1, T_max + 1)):
                z_t = self.sample_next_hidden_state(batch_mask, t, *self.transition_model(z_prev))

                self.sample_observation(padded_sequence, batch_mask, t, *self.observation_model(z_t))

                z_prev = z_t

    def sample_observed_time_series(self,batch):
        unconditioned_model = trace(uncondition(self.time_series_model))
        # batch doesnt matter, only for shape infering

        unconditioned_model(*batch)

        filter_observations = lambda pair: pair[0].startswith(self.OBSERVED_VARIABLE_NAME + "_")

        obs_nodes = dict(filter(filter_observations, unconditioned_model.msngr.trace.nodes.items()))
        values = [x["value"] for x in obs_nodes.values()]
        time_series = torch.stack(values, 1)
        return time_series