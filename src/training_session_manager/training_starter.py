from typing import Tuple

from hydra.utils import instantiate
from lightning import Trainer, LightningModule, LightningDataModule
from omegaconf import DictConfig

from src.training_session_manager.training_session_manager import TrainingSessionManager
from src.lightning_module import LightningVariationalHiddenMarkov


class TrainingStarter(TrainingSessionManager):

    def __call__(self, cfg: DictConfig) -> Tuple[Trainer, LightningModule, LightningDataModule]:
        module = instantiate(cfg)
        lightning_module: LightningVariationalHiddenMarkov = module.lightning_module
        trainer: Trainer = module.trainer
        data_module = module.data_module

        return trainer, lightning_module, data_module
