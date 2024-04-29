import torch
import torch.nn as nn
import pyro
from typing import Type, Callable

from src.utils.variable_time_series_length_utils import collate_fn_2
from pyro.poutine import trace, uncondition
from typing import TypeVar, TYPE_CHECKING, Iterable, Optional
from src.utils.custom_typehint import TensorIterable
from lightning import LightningModule
if TYPE_CHECKING:
    from pyro.distributions import TorchDistributionMixin

D_O = TypeVar("D_O", bound="TorchDistributionMixin")
D_H = TypeVar("D_H", bound="TorchDistributionMixin")


class TimeSeriesModel(LightningModule):
    HIDDEN_VARIABLE_NAME = "z"
    OBSERVED_VARIABLE_NAME = "x"

    def __init__(self, transition_model: nn.Module,
                 observation_model: nn.Module,
                 observation_distribution: Type[D_O],
                 transition_distribution: Type[D_H],
                 collate_fn: Callable[[TensorIterable], tuple[torch.Tensor, torch.Tensor]] = collate_fn_2, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_model = transition_model
        self.observation_model = observation_model

        self.transition_distribution = transition_distribution
        self.observation_distribution = observation_distribution
        self.collate_fn = collate_fn
        self.z_0 = nn.Parameter(torch.zeros(self.transition_model.z_dim))

    
    @staticmethod
    def get_mask(t, batch_mask = None) -> Optional[torch.Tensor]:
        return batch_mask[:,t-1] if batch_mask is not None else batch_mask
    def sample_observation(self, t, batch=None, batch_mask=None, *distribution_parameters):
        mask =  self.get_mask(t,batch_mask)
        distribution_instance = self.get_distribution_instance(self.observation_distribution, mask, *distribution_parameters)
            
        observation_name = f"{self.OBSERVED_VARIABLE_NAME}_{t}"
        obs_t = batch[:, t - 1, :] if batch is not None else None
        obs_t = pyro.sample(observation_name, distribution_instance.to_event(1), obs=obs_t)
        return obs_t
    
    @staticmethod
    def get_distribution_instance(distribution: Type["TorchDistributionMixin"], mask=None, *distribution_parameters: torch.Tensor) -> "TorchDistributionMixin":
        distribution_instance = distribution(*distribution_parameters)
        if mask is not None:
            distribution_instance = distribution_instance.mask(mask)
        
        return distribution_instance

    def sample_next_hidden_state(self, t: int, batch_mask: Optional[torch.Tensor] =None, *distribution_parameters: torch.Tensor) -> torch.Tensor:
        hidden_state_name = f"{self.HIDDEN_VARIABLE_NAME}_{t}"
        mask = self.get_mask(t,batch_mask)
        distribution_instance = self.get_distribution_instance(self.transition_distribution, mask, *distribution_parameters)
        
        z_t = pyro.sample(hidden_state_name, distribution_instance.to_event(1))
        return z_t

    def __call__(self, batch: TensorIterable)-> torch.Tensor:
        pyro.module("time_series_model", self)

        padded_sequence, batch_mask = self.collate_fn(batch)
        t_max = padded_sequence.size(1)
        time_range = range(1, t_max + 1)
        n_batches = padded_sequence.size(0)

        z_prev = self.z_0.repeat(n_batches, 1).to(self.z_0)


        z_prev = self.run_over_time_range(z_prev, time_range, batch_mask, padded_sequence)

        return z_prev

    def run_over_time_range(self, z_prev: torch.Tensor, time_range: Iterable[int],
                            batch_mask: Optional[torch.Tensor] = None,
                            padded_sequence: Optional[torch.Tensor] = None) -> torch.Tensor:


        for t in pyro.markov(time_range):
            z_prev = self.run_step(t, z_prev, padded_sequence, batch_mask)

        return z_prev

    def run_step(self, t: int, z_prev: torch.Tensor, padded_sequence: Optional[torch.Tensor] = None, batch_mask: Optional[torch.Tensor] = None):
        distribution_parameters = self.transition_model(z_prev)
        z_t = self.sample_next_hidden_state(t, batch_mask, *distribution_parameters)

        observation_distribution_parameters = self.observation_model(z_t)
        self.sample_observation(t, padded_sequence, batch_mask, *observation_distribution_parameters)

        return z_t

    def generate_time_series_from_batch(self, batch):
        unconditioned_model = trace(uncondition(self))
        # batch doesnt matter, only for shape infering

        unconditioned_model(batch)

        filter_observations = lambda pair: pair[0].startswith(self.OBSERVED_VARIABLE_NAME + "_")

        obs_nodes = dict(filter(filter_observations, unconditioned_model.msngr.trace.nodes.items()))
        values = [x["value"] for x in obs_nodes.values()]
        time_series = torch.stack(values, 1)
        return time_series

