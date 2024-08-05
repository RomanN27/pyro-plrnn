from typing import Any

from lightning import LightningModule
from src.metrics.metric_base import MetricLogType, Metric
from src.models.hidden_markov_model import HiddenMarkovModel
import torch
class GradMetric(Metric):
    #not functional
    log_types = [MetricLogType.scalar_dict]
    def __repr__(self):
        return f"GradMetric"

    log_types = [MetricLogType.scalar_dict]

    def __init__(self):
        super().__init__()
        self.add_state()

    def update(self, hmm: HiddenMarkovModel) -> None:

        grads = self.grad_norm(hmm, 2)
        self.grads = {k: v.tolist() for k, v in grads.items()}
