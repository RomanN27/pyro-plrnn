from src.training_session_manager.training_session_manager import TrainingSessionManager
from omegaconf import DictConfig
from typing import Tuple
from lightning import Trainer, LightningModule, LightningDataModule
import logging
from copy import  deepcopy

class Retry(TrainingSessionManager):

    def __init__(self, max_retry: int, mgr: TrainingSessionManager):
        self.max_retry = max_retry
        self.mgr = mgr
        self.current_try = 0

    def __call__(self, cfg: DictConfig) -> Tuple[Trainer, LightningModule, LightningDataModule]:
        trainer, lightning_module, data_module = self.mgr(cfg)

        orig_fit = trainer.fit
        self.current_lightning_module = lightning_module

        def retry_fit(self_, *args,**kwargs):

            while self.current_try < self.max_retry:
                try:
                    orig_fit(model = self.current_lightning_module, datamodule=data_module)
                    break

                except Exception as e :
                    logging.error(f"Training Start Failed after try: {self.current_try} because of {type(e).__name__}")
                    new_trainer, new_lightning_module, _ = self.mgr(cfg)

                    self.current_lightning_module = new_lightning_module
                    self.current_try += 1

        trainer.fit = retry_fit

        return trainer, lightning_module, data_module
