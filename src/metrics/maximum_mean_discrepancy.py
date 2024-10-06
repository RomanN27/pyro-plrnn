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

        X = x_gen.flatten(end_dim=-2)
        Y = batch.flatten(end_dim=-2)

        xx, yy, zz = torch.mm(X, X.t()), torch.mm(Y, Y.t()), torch.mm(X, Y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)


        k_XX = self.kernel(dxx)
        k_XY = self.kernel(dxy)
        k_YY = self.kernel(dyy)

        m = X.size(0)
        norming_factor = m * (m - 1)

        mmd = k_XX.sum() / norming_factor + k_YY.sum() / norming_factor - 2 * k_XY.mean()

        p = 1/(self.n_times_updated + 1)
        self.mmd = (1-p) * self.mmd + p * mmd

        self.n_times_updated += 1



    def kernel(self,x:torch.Tensor):
        return torch.exp(- x / (2*self.bandwidth**2) )
    def compute(self):
        return self.mmd

