from typing import Any
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
@hydra.main(version_base=None,config_path="conf", config_name="config")
def app(cfg: DictConfig) -> Any:
    rnn = instantiate(cfg.guide)
    print(rnn)

app()
import pyro.infer.autoguide