from src.metrics.metric_base import Metric, MetricLogType
from src.models.hidden_markov_model import HiddenMarkovModel
from src.utils.lightning_utils import update_mean
from src.models.transition_models.plrnns.plrnn_base import PLRNN
from src.models.cov_mixins import ConstantCovarianceMixin

import torch
class SigmaMetric(Metric):
    log_types = [MetricLogType.scalar]

    def __init__(self):
        super().__init__()
        self.add_state("sigma", default=torch.tensor(0.))
        self.add_state("n", default=torch.tensor(0,dtype=torch.int32))

    def update(self, hmm: HiddenMarkovModel) -> None:
        cov_model: ConstantCovarianceMixin = hmm.transition_sampler.model

        sigma = cov_model.sigma.detach()
        self.sigma = update_mean(self.sigma, sigma, self.n)
        self.n += 1

    def compute(self) -> torch.Tensor:
        return self.sigma