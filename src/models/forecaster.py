from src.models.hidden_markov_model import HiddenMarkovModel
from pyro.poutine import trace, replay
from pyro import plate
from typing import Callable
import torch
from pyro.poutine import Trace
from pyro.contrib.autoname import scope
from pyro.poutine.runtime import Message
from typing import Any
from src.utils.trace_utils import get_observed_values_from_trace, get_hidden_values_from_trace
from src.utils.variable_group_enum import V

class Forecaster:
    PRED_PREFIX = "pred"

    def __init__(self, model: HiddenMarkovModel, guide: Callable[[torch.Tensor], Any]) -> None:
        self.model = model
        self.guide = guide

    def __call__(self, batch: torch.Tensor, n_timesteps_to_forecast: int, n_samples: int, probabilistic: bool = True, truncate_batch: bool = False) -> torch.Tensor:

        guide_trace = self.get_guide_trace(batch, n_samples)
        posterior_model_trace = self.run_model_over_posterior_distribution(batch, guide_trace, n_timesteps_to_forecast, n_samples)

        predictive_tensor = self.process_trace(posterior_model_trace)
        #predictive tensor shape: n_samples * n_batches * n_time_steps * n_dim

        if not probabilistic:
            predictive_tensor = [t.mean(0) for t in predictive_tensor]

        return predictive_tensor



    def get_guide_trace(self, batch: torch.Tensor, n_samples: int) -> Trace:
        traced_guide = trace(self.guide)

        with plate("_num_posterior_samples", n_samples,dim=-2):
            traced_guide(batch)

        return traced_guide.trace

    def run_model_over_posterior_distribution(self, batch: torch.Tensor, guide_trace: Trace, delta_t: int,
                                              n_samples: int) -> Trace:
        posterior_model = replay(self.model, trace=guide_trace)
        t_prediction_start = batch.size(1) + 1
        t_prediction_end =  t_prediction_start + delta_t
        time_range = range(t_prediction_start, t_prediction_end)


        posterior_trace = trace(posterior_model).get_trace(n_samples)
        z_1 = get_hidden_values_from_trace(posterior_trace)
        with trace() as tracer:
            with scope(prefix=self.PRED_PREFIX):
                self.model.run_over_time_range(z_prev=z_1,time_range= time_range)

        return tracer.trace

    @staticmethod
    def get_values_from_nodes(nodes: list[Message]) -> torch.Tensor:
        return [node["value"] for node in nodes]

    def process_trace(self, posterior_model_trace: Trace) -> torch.Tensor:


        inputed_observed_nodes = [posterior_model_trace.nodes[obs_node] for obs_node in
                                  posterior_model_trace.observation_nodes]
        inputed_observed_values = self.get_values_from_nodes(inputed_observed_nodes)
        inputed_observed_values = torch.stack(inputed_observed_values)
        inputed_observed_values = inputed_observed_values.transpose(0, 1)

        is_pred_node = lambda node: node.startswith(f"{self.PRED_PREFIX}/{self.model.OBSERVED_VARIABLE_NAME}")
        pred_nodes = [posterior_model_trace.nodes[node] for node in posterior_model_trace.stochastic_nodes if
                      is_pred_node(node)]
        pred_values = self.get_values_from_nodes(pred_nodes)
        pred_values = torch.stack(pred_values)
        pred_values = pred_values.swapaxes(0,1).swapaxes(1,2)
        n_mc_samples = pred_values.size(0)
        inputed_observed_values = inputed_observed_values.unsqueeze(0).expand(n_mc_samples, -1, -1, -1)

        actually_observed_list = [b for b in inputed_observed_values.swapaxes(0,1)] # lengths)]

        all_values = [torch.cat([actual_batch, predicted_batch ], dim =-2) for actual_batch, predicted_batch in zip(actually_observed_list,pred_values.swapaxes(0,1) )]

        return all_values


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from src.models.forecaster import Forecaster

    path = r"/Users/romannefedov/PycharmProjects/pyro-plrnn/conf/flattened_configs/flattened_config_3.yaml"

    with open(path, "r") as f:
        cfg = OmegaConf.load(f)

    module = instantiate(cfg)

    forecaster: Forecaster = module.lightning_module.forecaster
    data = module.data_module.dataset[1].unsqueeze(0)

    forecasts = forecaster(data, 100, 100)
