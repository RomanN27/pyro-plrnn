import random
from typing import Any

import torch
from src.metrics.metric_base import Metric, MetricLogType

from src.models.hidden_markov_model import HiddenMarkovModel

class GaussianMaximumMeanDiscrepancy(Metric):
    log_types = [MetricLogType.scalar]

    def __init__(self, bandwidth: float,n_samples: int, n_time_steps: int):
        super().__init__()
        self.add_state("mmd", default=torch.tensor(0))
        self.add_state("n_times_updated", default=torch.tensor(0,dtype=torch.int32))
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        self.n_time_steps = n_time_steps


    def update(self, hmm: HiddenMarkovModel, batch: torch.Tensor) -> None:
        n_samples = self.n_samples if self.n_samples >= 0 else len(batch)
        n_time_steps = self.n_time_steps if self.n_time_steps >= 0 else batch.size(-2)

        empty_batch = torch.empty(n_samples, n_time_steps, batch.size(-1))



        _, x_gen = hmm.get_history(empty_batch)

        x_gen_flatten = x_gen.flatten(end_dim=-2)
        x_true_flatten = batch.flatten(end_dim=-2)

        mmd = self.kernel(self.get_scalar_prod(x_gen_flatten,x_true_flatten)).mean()

        p = 1/(self.n_times_updated + 1)
        self.mmd = (1-p) * self.mmd + p * mmd

        self.n_times_updated += 1


    def get_scalar_prod(self, X, Y):
        XX = (X**2).sum(1)
        YY = (Y**2).sum(1)
        XY = X @ Y.T
        diff_scalar_product = XX.reshape(-1,1) - 2 * XY.T + YY.reshape(1, -1)
        return diff_scalar_product


    def kernel(self,x:torch.Tensor):
        return torch.exp(- x / (2*self.bandwidth**2) )
    def compute(self):
        return self.mmd

