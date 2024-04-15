from time_series_model import TimeSeriesModel
from pyro.poutine import trace, replay
from pyro import plate
from typing import Callable
import torch
from custom_typehint import TensorIterable
from pyro.poutine import Trace
from pyro.contrib.autoname import scope
from pyro.poutine.runtime import Message
from typing import Any


class Forecaster:
    PRED_PREFIX = "pred"

    def __init__(self, model: TimeSeriesModel, guide: Callable[[TensorIterable], Any]) -> None:
        self.model = model
        self.guide = guide

    def __call__(self, batch: TensorIterable, t: int, n_samples: int, probabilistic: bool = True) -> torch.Tensor:
        guide_trace = self.get_guide_trace(batch, n_samples)
        posterior_model_trace = self.run_model_over_posterior_distribution(batch, guide_trace, t, n_samples)

        predictive_tensor = self.process_trace(posterior_model_trace)

        if not probabilistic:
            predictive_tensor = predictive_tensor.mean(0)

        return predictive_tensor

    def get_guide_trace(self, batch: TensorIterable, n_samples: int) -> Trace:
        traced_guide = trace(self.guide)

        with plate("_num_predictive_samples", n_samples):
            traced_guide(batch)

        return traced_guide.trace

    def run_model_over_posterior_distribution(self, batch: TensorIterable, guide_trace: Trace, t: int,
                                              n_samples: int) -> Trace:
        posterior_model = replay(self.model, trace=guide_trace)
        t_0 = batch[0].size(0) + 1
        time_range = range(t_0, t_0 + t)

        with trace() as tracer:
            with plate("_num_predictive_samples", n_samples):
                z_h = posterior_model(batch)
                with scope(prefix=self.PRED_PREFIX):
                    self.model.run_over_time_range(z_h, time_range)

        return tracer.trace

    @staticmethod
    def get_values_from_nodes(nodes: list[Message]) -> TensorIterable:
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
        pred_values = pred_values.transpose(0, -2)
        n_mc_samples = pred_values.size(0)
        inputed_observed_values = inputed_observed_values.unsqueeze(0).expand(n_mc_samples, -1, -1, -1)

        all_values = torch.cat([inputed_observed_values, pred_values], dim=-2)

        return all_values
