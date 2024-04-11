from trainer import AnnealingTrainer, Trainer
import hydra

from omegaconf import DictConfig, OmegaConf
import torch
import mlflow
from torchinfo import summary

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    cfg, trainer = get_and_load_trainer(cfg)

    with mlflow.start_run(nested=True) as run:
        log_to_mlflow(trainer, cfg)
        print(run.info.run_id)
        trainer.train(**cfg.training)

        artifact_path = artifact_path_from_uri(run.info.artifact_uri)
        trainer.save(artifact_path + "/model.pt")

def log_to_mlflow(trainer, cfg):
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(trainer.time_series_model)))
    mlflow.log_artifact("model_summary.txt")

    mlflow.set_tag("Training Info", "Just trying mlflow a bit")
    params = OmegaConf.to_yaml(cfg, resolve=True)
    mlflow.log_params({"config": params})

def get_and_load_trainer(cfg):
    if run_id := cfg.get("run_id"):
        training_cfg = Trainer.get_config_from_run_id(run_id)
        cfg = OmegaConf.merge(training_cfg, cfg)
        mlflow_client = mlflow.tracking.MlflowClient()
        artifact_uri = mlflow_client.get_run(run_id=run_id).info.artifact_uri
        artifact_path = artifact_path_from_uri(artifact_uri)
    trainer = AnnealingTrainer.get_trainer_from_config(cfg)
    if run_id:
        trainer.load(artifact_path + "/model.pt")
    return cfg, trainer

def artifact_path_from_uri(uri:str)->str:
    return uri.replace("mlflow-artifacts:", "mlartifacts")

if __name__ == "__main__":
    main()
