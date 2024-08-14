from typing import Tuple, Literal, Optional

from hydra.utils import instantiate
from lightning import Trainer, LightningModule, LightningDataModule
from omegaconf import DictConfig, OmegaConf

from src.training_session_manager.training_session_manager import TrainingSessionManager
from src.lightning_modules.variational_lightning_module import VariationalTeacherForcing
from src.utils.lightning_utils import update_state_dict
from src.utils.mlflow_utils import get_config_from_run_id, get_checkpoint_from_run_id

ModelComponents = list[Literal["hidden_markov_model", "variational_distribution"]]


class TrainingRestarter(TrainingSessionManager):
    def __init__(self, run_id: str, models_to_reload: Optional[ModelComponents] = None, overwrite_cfg: bool = True):
        self.run_id = run_id
        self.models_to_reload = models_to_reload or []
        self.overwrite_cfg = overwrite_cfg

    def __call__(self, cfg: DictConfig) -> Tuple[Trainer, LightningModule, LightningDataModule]:
        run_cfg = get_config_from_run_id(self.run_id)
        if self.overwrite_cfg:
            merged_cfg = OmegaConf.merge(run_cfg, cfg)
            module = instantiate(merged_cfg)
        else:
            module = instantiate(run_cfg)

        state_dict_to_use = self.get_relevant_state_dict()

        lightning_module: VariationalTeacherForcing = module.lightning_module
        update_state_dict(lightning_module, state_dict_to_use)

        trainer: Trainer = module.trainer

        data_module = module.data_module

        return trainer, lightning_module, data_module

    def get_relevant_state_dict(self):
        ckpt = get_checkpoint_from_run_id(self.run_id)
        state_dict = ckpt["state_dict"]
        state_dict_to_use = {k: v for k, v in state_dict.items() if k.split(".")[0] in self.models_to_reload}
        return state_dict_to_use
