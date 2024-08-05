import torch
import torch.nn as nn
import pyro
from typing import Type, Callable, Generic, Optional
from pyro.distributions import MultivariateNormal
from src.utils.variable_time_series_length_utils import collate_fn_2
from pyro.poutine import trace, uncondition, condition, Trace
from pyro import plate
from typing import TypeVar, TYPE_CHECKING, Iterable, Optional, Tuple
from src.utils.custom_typehint import TensorIterable
from lightning import LightningModule
from src.utils.trace_utils import get_hidden_values_from_trace, get_observed_values_from_trace

if TYPE_CHECKING:
    from pyro.distributions import TorchDistributionMixin, TorchDistribution, Distribution
    from src.models.model_sampler import ModelBasedSampler, ModelType, DistType
    from src.data.data_module import DataType, DatasetType, DataLoader
from src.utils.variable_group_enum import V

ObservationModelType = TypeVar("ObservationModelType", bound=nn.Module)
LatentModelType = TypeVar("LatentModelType", bound=nn.Module)


class HiddenMarkovModel(nn.Module, Generic[LatentModelType, ObservationModelType]):



    def __init__(self, transition_sampler: "ModelBasedSampler[torch.Tensor,LatentModelType,DistType]",
                 observation_sampler: "ModelBasedSampler[torch.Tensor,ObservationModelType,DistType]",
                 initial_sampler: "ModelBasedSampler[None,ModelType,DistType]",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_sampler = transition_sampler
        self.observation_sampler = observation_sampler
        self.initial_sampler = initial_sampler

    def __call__(self,batch: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        n_samples = batch.size(0)
        n_time_steps = batch.size(-2)

        pyro.module("hidden_markov_model", self) #TODO find out if this is needed


        Z_1 = self.run_first_step(n_samples)

        self.run_over_time_range(Z_1, range(2, n_time_steps + 1))

    def get_history(self,batch) -> Tuple[torch.Tensor,torch.Tensor]:
        trace_ = trace(self).get_trace(batch)
        z_history = get_hidden_values_from_trace(trace_)
        x_history = get_observed_values_from_trace(trace_)

        z_tensor = torch.stack(z_history, -2)
        x_tensor = torch.stack(x_history, -2)

        return z_tensor, x_tensor

    def run_first_step(self, n_samples: int) -> torch.Tensor:
        with plate("n_samples", n_samples, dim=-2):
            z_1 = self.initial_sampler(None, name=f"{V.LATENT}_1")
        x_1 = self.emit(z_1, 1)
        return z_1

    def run_over_time_range(self, z_prev: torch.Tensor, time_range: Iterable[int]) -> torch.Tensor:

        for t in pyro.markov(time_range):
            z_prev = self.run_step(t, z_prev)

        return z_prev

    def run_step(self, t: int, z_prev: torch.Tensor )-> torch.Tensor:
        z_t = self.transition(z_prev, t)

        self.emit(z_t,t)

        return z_t

    def transition(self, z_prev: torch.Tensor, t: int):
        hidden_state_name = f"{V.LATENT}_{t}"
        z_t = self.transition_sampler(z_prev, hidden_state_name)
        return z_t

    def emit(self, z_t: torch.Tensor, t:int):
        observation_name = f"{V.OBSERVED}_{t}"
        x_t = self.observation_sampler(z_t, observation_name)
        return x_t


    @classmethod
    def get_observations_from_trace(cls,trace:Trace):
        filter_observations = lambda pair: pair[0].startswith(V.OBSERVED + "_")

        obs_nodes = dict(filter(filter_observations, trace.nodes.items()))
        values = [x["value"] for x in obs_nodes.values()]
        time_series = torch.stack(values, 1)
        return time_series
