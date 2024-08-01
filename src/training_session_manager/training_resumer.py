from typing import Tuple

from hydra.utils import instantiate
from lightning import Trainer, LightningModule, LightningDataModule
from omegaconf import DictConfig

from src.training_session_manager.training_session_manager import TrainingSessionManager

from src.lightning_modules.variational_lightning_module import LightningVariationalHiddenMarkov
from src.utils.mlflow_utils import get_config_from_run_id, get_ckpt_path_from_run_id, get_checkpoint_from_run_id
from functools import partial

class TrainingResumer(TrainingSessionManager):

    def __init__(self, run_id: str):
        self.run_id = run_id

    def __call__(self, cfg: DictConfig) -> Tuple[Trainer, LightningModule, LightningDataModule]:
        run_cfg = get_config_from_run_id(self.run_id)
        run_cfg.parent_run_id = self.run_id

        module = instantiate(run_cfg)

        check_point_name = cfg.get("check_point_name", None)
        ckpt_path = get_ckpt_path_from_run_id(self.run_id, check_point_name=check_point_name)
        ckpt = get_checkpoint_from_run_id(self.run_id, check_point_name)

        lightning_module: LightningVariationalHiddenMarkov = module.lightning_module
        lightning_module.load_state_dict(ckpt["state_dict"])

        trainer = module.trainer
        self.load_training_state(trainer, ckpt_path=ckpt_path)

        return trainer, lightning_module, module.data_loader


    @staticmethod
    def load_training_state(trainer: Trainer, ckpt_path) -> None:
        old_fit = trainer.fit
        trainer.fit = partial(old_fit, ckpt_path=ckpt_path)