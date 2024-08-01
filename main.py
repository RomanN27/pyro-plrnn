import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.training_session_manager import TrainingSessionManager, TrainingStarter
from src.utils.hydra_utils import listify_config
from lightning import Trainer, LightningModule, LightningDataModule
import matplotlib.pyplot as plt
plt.ioff()

@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig):
    cfg = listify_config(cfg)

    training_session_manager: TrainingSessionManager = instantiate(cfg.training_session_manager)

    trainer: Trainer
    lightning_module: LightningModule
    data_module: LightningDataModule

    trainer, lightning_module, data_module = training_session_manager(cfg)




    trainer.logger.log_hyperparams(cfg)

    trainer.fit(lightning_module, datamodule=data_module)


    trainer.test(lightning_module, datamodule=data_module)


if __name__ == '__main__':
    main()
