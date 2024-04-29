from src.lightning_modules import AnnealingModule, TimeSeriesModule
import hydra
from lightning.pytorch.loggers import MLFlowLogger

from omegaconf import DictConfig, OmegaConf
import mlflow
from torchinfo import summary
from lightning.pytorch import Trainer as LightningTrainer

mlflow_logger = MLFlowLogger()


@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig):


    cfg, trainer = get_and_load_trainer(cfg)

    data_module = trainer.data_loader

    lightning_trainer = LightningTrainer(logger=mlflow_logger,num_sanity_val_steps=0,accelerator="cpu")
    lightning_trainer.fit(trainer,datamodule=data_module)




    #with mlflow.start_run(nested=True) as run:
    #    log_to_mlflow(trainer, cfg)
    #    print(run.info.run_id)
    #    trainer.train(**cfg.training)

        #artifact_path = artifact_path_from_uri(run.info.artifact_uri)
       # trainer.save(artifact_path + "/model.pt")

def log_to_mlflow(trainer, cfg):
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(trainer.time_series_model)))
    mlflow.log_artifact("model_summary.txt")

    mlflow.set_tag("Training Info", "Just trying mlflow a bit")
    params = OmegaConf.to_yaml(cfg, resolve=True)
    mlflow.log_params({"config": params})

def get_and_load_trainer(cfg):
    if run_id := cfg.get("run_id"):
        training_cfg = TimeSeriesModule.get_config_from_run_id(run_id)
        cfg = OmegaConf.merge(training_cfg, cfg)
        mlflow_client = mlflow.tracking.MlflowClient()
        artifact_uri = mlflow_client.get_run(run_id=run_id).info.artifact_uri
        artifact_path = artifact_path_from_uri(artifact_uri)
    trainer = AnnealingModule.get_trainer_from_config(cfg)
    if run_id:
        trainer.load(artifact_path + "/model.pt")
    return cfg, trainer

def artifact_path_from_uri(uri:str)->str:
    return uri.replace("mlflow-artifacts:", "mlartifacts")

if __name__ == "__main__":
    main()
