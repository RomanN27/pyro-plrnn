from trainer import get_trainer_from_config
import hydra

from omegaconf import DictConfig, OmegaConf
import torch
import mlflow
from torchinfo import summary
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

@hydra.main(version_base=None,config_path="conf", config_name="config")
def main(cfg: DictConfig):
    #mlflow.set_experiment("MLflow Quickstart")


    print(OmegaConf.to_yaml(cfg,resolve=True))

    trainer = get_trainer_from_config(cfg)

    with mlflow.start_run(nested=True) as run:

        with open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(trainer.time_series_model)))

        mlflow.log_artifact("model_summary.txt")

        mlflow.set_tag("Training Info", "Just trying mlflow a bit")
        params = OmegaConf.to_yaml(cfg,resolve=True)
        mlflow.log_params({"config":params})
        trainer.train(cfg.training.n_epochs,
                      cfg.training.annealing_factor,
                      cfg.training.annealing_epochs)

        artificat_path = run.info.artifact_uri.replace("mlflow-artifacts:", "mlartifacts")
        torch.save(trainer.time_series_model.state_dict(), artificat_path + "/time_series_model.pt")
        torch.save(trainer.variational_distribution.state_dict(), artificat_path + "/variational_model.pt")


if __name__ == "__main__":
    main()

