from abc import ABC, abstractmethod
from lightning import LightningModule, LightningDataModule
from typing import Tuple

from lightning import Trainer
from omegaconf import DictConfig


class TrainingSessionManager(ABC):
    """
       Abstract base class for managing the configuration and state of a Lightning training process.

       This class defines the interface for setting up and restarting training sessions in PyTorch Lightning.
       Implementations of this class are expected to provide mechanisms for instantiating and returning
       a Trainer, LightningModule, and LightningDataModule based on a given configuration.

       The class integrates Hydra for configuration management, PyTorch Lightning for the training process,
       and MLflow for experiment tracking and state management. This allows for flexible and reproducible
       training workflows.

       Methods
       -------
       __call__(cfg: DictConfig) -> Tuple[Trainer, LightningModule, LightningDataModule]:
           Abstract method to instantiate and return the Trainer, LightningModule, and LightningDataModule
           based on the provided configuration.
       """
    @abstractmethod
    def __call__(self, cfg: DictConfig) -> Tuple[Trainer, LightningModule, LightningDataModule]:
        pass


