from typing import Literal

from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from src.lightning_module import LightningVariationalHiddenMarkov
from src.utils.lightning_utils import update_state_dict
from src.utils.mlflow_utils import get_config_from_run_id, get_ckpt_path_from_run_id, get_checkpoint_from_run_id

ModelComponents = list[Literal["hidden_markov_model", "variational_distribution"]]
def resume_training(cfg: DictConfig):

    run_id = cfg.run_id
    run_cfg = get_config_from_run_id(run_id)
    run_cfg.run_id = run_id

    module = instantiate(run_cfg)


    check_point_name = cfg.get("check_point_name", None)
    ckpt_path = get_ckpt_path_from_run_id(run_id,check_point_name=check_point_name)
    ckpt = get_checkpoint_from_run_id(run_id,check_point_name)

    lightning_module: LightningVariationalHiddenMarkov = module.lightning_module
    lightning_module.load_state_dict(ckpt["state_dict"])

    trainer = module.trainer
    trainer.logger.log_hyperparams(run_cfg)
    trainer.fit(lightning_module, datamodule=module.data_loader, ckpt_path = ckpt_path)


def restart_training(cfg:DictConfig):
    run_id = cfg.run_id
    run_cfg = get_config_from_run_id(run_id)
    merged_cfg = OmegaConf.merge(run_cfg, cfg)

    weights_to_reload: ModelComponents = cfg.weights_to_reload


    ckpt = get_checkpoint_from_run_id(run_id)
    state_dict = ckpt["state_dict"]
    state_dict_to_use = {k:v for k,v in state_dict.items() if k.split(".")[0] in weights_to_reload}

    module = instantiate(merged_cfg)

    lightning_module: LightningVariationalHiddenMarkov = module.lightning_module
    update_state_dict(lightning_module, state_dict_to_use)

    trainer: Trainer = module.trainer

    trainer.logger.log_hyperparams(merged_cfg)

    data_module = module.data_loader

    trainer.fit(lightning_module, datamodule=data_module)


def start_new_training(cfg: DictConfig):
    module = instantiate(cfg)
    lightning_module: LightningVariationalHiddenMarkov = module.lightning_module
    trainer: Trainer = module.trainer
    trainer.logger.log_hyperparams(cfg)
    data_module = module.data_loader
    trainer.fit(lightning_module, datamodule=data_module)



