import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.training_session_manager import TrainingSessionManager, TrainingStarter
from src.utils.hydra_list import listify_config


@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig):
    cfg = listify_config(cfg)

    training_session_manager: TrainingSessionManager = instantiate(cfg.training_session_manager)

    trainer, lightning_module, data_module = training_session_manager(cfg)

    trainer.logger.log_hyperparams(cfg)

    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == '__main__':
    main()
