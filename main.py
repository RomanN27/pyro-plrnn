import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from src.lightning_module import LightningVariationalHiddenMarkov

#mlflow_logger = MLFlowLogger()
@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig):
    lightning_module: LightningVariationalHiddenMarkov = instantiate(cfg)
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


    path = r"C:\Users\roman.nefedov\PycharmProjects\PLRNN_Family_Variational_Inference\plots\checkpoints"
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="dsr_loss")
    early_stopping = EarlyStopping(monitor="dsr_loss", mode="min")
    callbacks = [checkpoint_callback,early_stopping]
    lightning_trainer = LightningTrainer(callbacks=callbacks, enable_checkpointing=True, num_sanity_val_steps=0,
                                         accelerator="cpu", max_epochs=10000, default_root_dir=path,min_epochs=15)

    lightning_trainer.fit(lightning_module, datamodule=lightning_module.data_loader)


if __name__ == '__main__':
    main()