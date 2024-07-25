import hydra
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import sys
from pathlib import Path
mlflow_logger = MLFlowLogger()

FLATTENED_DIRECTORY_NAME = "flattened_configs"
@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig):
    hydra_config = HydraConfig.get()
    cfg_source = hydra_config.runtime.config_sources
    cfg_main_source = next(filter(lambda x: x["provider"] =="main" , cfg_source))
    cfg_main_source_path = Path(cfg_main_source["path"])

    cfg_name = hydra_config.job.config_name
    cfg_flattened_name = "flattened_" + cfg_name

    flattened_config_path = cfg_main_source_path / FLATTENED_DIRECTORY_NAME / cfg_flattened_name

    with open(flattened_config_path,"w") as f:
        OmegaConf.save(cfg,f)



if __name__ == '__main__':
    main()