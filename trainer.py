import tqdm
from hydra.utils import instantiate
from pyro.poutine import scale
from torch.utils.data import Dataset, Sampler, RandomSampler
from typing import Callable
from pyro.optim import PyroOptim
from pyro.infer import ELBO, SVI
from dataclasses import dataclass
from tqdm import tqdm
import mlflow
import re
from time_series_model import TimeSeriesModel
from typing import TYPE_CHECKING
from selective_scale_messenger import SelectiveScaleMessenger
if TYPE_CHECKING:
    from pyro.poutine.runtime import Message
from omegaconf import DictConfig

@dataclass
class TrainingConfig:
    annealing_factor: float = 1.0
    annealing_epochs: int = 3


class AnnealingTimeSeriesTrainer:

    def __init__(self, time_series_model: TimeSeriesModel, variational_distribution: Callable,
                 data_loader: Dataset, optimizer: PyroOptim, elbo: ELBO):

        self.time_series_model = time_series_model
        self.variational_distribution = variational_distribution
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.elbo = elbo



        self.svi = SVI(self.time_series_model, variational_distribution, self.optimizer, loss=self.elbo)

    def annealing_selector(self, msg: "Message") -> bool:
        z_name = self.time_series_model.HIDDEN_VARIABLE_NAME
        pattern = rf"^{z_name}_\d+"
        name = msg["name"]
        return  name and bool(re.match(pattern, name))


    def train(self, n_epochs: int, min_af: int, annealing_epochs: int):

        step = 0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            annealing_factor = self.get_annealing_factor(annealing_epochs, epoch, min_af)

            with SelectiveScaleMessenger(annealing_factor, self.annealing_selector):
                for batch in tqdm(self.data_loader):
                    loss = self.svi.step(batch)
                    epoch_loss += loss
                    if epoch > 5:
                        mlflow.log_metric("loss", f"{loss:2f}", step=step)
                    step += 1
                print(epoch_loss)

    def get_annealing_factor(self, annealing_epochs, epoch, min_af) -> float:
        annealing_factor = min_af + (1 - min_af) * min(1, epoch / annealing_epochs)
        return  annealing_factor

    def save_checkpoint(self):
        pass


def get_trainer_from_config(cfg: DictConfig) -> AnnealingTimeSeriesTrainer:
    train,test,valid = instantiate(cfg.data)
    plrnn = instantiate(cfg.transition_model)
    observation_model = instantiate(cfg.observation_model)
    observation_distribution = instantiate(cfg.observation_distribution)
    transition_distribution = instantiate(cfg.transition_distribution)
    time_series_model = TimeSeriesModel(plrnn, observation_model, observation_distribution, transition_distribution)

    optimizer_class = instantiate(cfg.optimizer.optimizer_class)
    optimizer = optimizer_class(dict(cfg.optimizer.optim_args))
    guide = instantiate(cfg.guide)
    loss = instantiate(cfg.loss)
    trainer = AnnealingTimeSeriesTrainer(time_series_model, guide, train, optimizer, loss)
    return trainer
