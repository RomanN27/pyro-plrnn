import torch
import torch.nn as nn
import pyro
from typing import Type, Callable

from src.utils.variable_time_series_length_utils import collate_fn_2
from pyro.poutine import trace, uncondition, condition, Trace
from typing import TypeVar, TYPE_CHECKING, Iterable, Optional
from src.utils.custom_typehint import TensorIterable
from lightning import LightningModule
if TYPE_CHECKING:
    from pyro.distributions import TorchDistributionMixin, TorchDistribution, Distribution
    from src.models.transition_models import ModelBasedSampler

D_O = TypeVar("D_O", bound="TorchDistributionMixin")
D_H = TypeVar("D_H", bound="TorchDistributionMixin")




class HiddenMarkovModel(LightningModule):
    HIDDEN_VARIABLE_NAME = "z"
    OBSERVED_VARIABLE_NAME = "x"

    def __init__(self, transition_model: ModelBasedSampler,
                 observation_model: ModelBasedSampler,
                  *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_model = transition_model
        self.observation_model = observation_model

        self.z_0 = nn.Parameter(torch.zeros(self.transition_model.z_dim))

    def __call__(self, batch: torch.Tensor)-> torch.Tensor:
        pyro.module("time_series_model", self)


        t_max = batch.size(1)
        time_range = range(1, t_max + 1)
        n_batches = batch.size(0)

        Z_0 = self.z_0.repeat(n_batches, 1).to(self.z_0)

        self.run_over_time_range(Z_0, time_range)


    def run_over_time_range(self, z_prev: torch.Tensor, time_range: Iterable[int]) -> torch.Tensor:

        for t in pyro.markov(time_range):
            z_prev = self.run_step(t, z_prev)

        return z_prev

    def run_step(self, t: int, z_prev: torch.Tensor ):
        hidden_state_name = f"{self.HIDDEN_VARIABLE_NAME}_{t}"
        z_t = self.transition_model(z_prev,hidden_state_name)

        observation_name = f"{self.OBSERVED_VARIABLE_NAME}_{t}"
        self.observation_model(z_t, observation_name)

        return z_t

    @classmethod
    def get_observations_from_trace(cls,trace:Trace):
        filter_observations = lambda pair: pair[0].startswith(cls.OBSERVED_VARIABLE_NAME + "_")

        obs_nodes = dict(filter(filter_observations, trace.nodes.items()))
        values = [x["value"] for x in obs_nodes.values()]
        time_series = torch.stack(values, 1)
        return time_series


class BayesianHiddenMarkovModel(HiddenMarkovModel):
    def __init__(self, transition_model: nn.Module,
                 observation_model: nn.Module,
                 observation_distribution: Type[D_O],
                 transition_distribution: Type[D_H],
                 transition_prior: Callable[[], "TorchDistributionMixin"],
                 z_0_prior: Callable[[], "TorchDistributionMixin"],
                 collate_fn: Callable[[TensorIterable], tuple[torch.Tensor, torch.Tensor]] = collate_fn_2, *args,
                 **kwargs):
        super().__init__(transition_model, observation_model, observation_distribution, transition_distribution, collate_fn, *args, **kwargs)
        self.transition_prior = transition_prior
        self.z_0_prior = z_0_prior

    def sample_transition_parameters(self, n_batches: int):
        prior_dist = self.transition_prior()
        sampled_params = pyro.sample("transition_params", prior_dist.expand([n_batches]).to_event(1))
        return sampled_params

    def sample_initial_state(self, n_batches: int):
        prior_dist = self.z_0_prior()
        z_0_sampled = pyro.sample("z_0", prior_dist.expand([n_batches, self.z_0.size(0)]).to_event(1))
        return z_0_sampled

    def __call__(self, batch: TensorIterable) -> torch.Tensor:
        pyro.module("time_series_model", self)
        padded_sequence, batch_mask = self.collate_fn(batch)
        t_max = padded_sequence.size(1)
        time_range = range(1, t_max + 1)
        n_batches = padded_sequence.size(0)

        # Sample transition parameters and initial state for each batch
        sampled_transition_params = self.sample_transition_parameters(n_batches)
        z_0_sampled = self.sample_initial_state(n_batches)

        z_prev = z_0_sampled.to(self.z_0)
        z_prev = self.run_over_time_range(z_prev, time_range, batch_mask, padded_sequence, sampled_transition_params)
        return z_prev

    def run_over_time_range(self, z_prev: torch.Tensor, time_range: Iterable[int],
                            batch_mask: Optional[torch.Tensor] = None,
                            padded_sequence: Optional[torch.Tensor] = None,
                            sampled_transition_params: torch.Tensor = None) -> torch.Tensor:
        for t in pyro.markov(time_range):
            z_prev = self.run_step(t, z_prev, padded_sequence, batch_mask, sampled_transition_params)
        return z_prev

    def run_step(self, t: int, z_prev: torch.Tensor, padded_sequence: Optional[torch.Tensor] = None, batch_mask: Optional[torch.Tensor] = None, sampled_transition_params: torch.Tensor = None):
        distribution_parameters = self.transition_model(z_prev, sampled_transition_params)
        z_t = self.sample_next_hidden_state(t, batch_mask, *distribution_parameters)
        observation_distribution_parameters = self.observation_model(z_t)
        self.sample_observation(t, padded_sequence, batch_mask, *observation_distribution_parameters)
        return z_t
