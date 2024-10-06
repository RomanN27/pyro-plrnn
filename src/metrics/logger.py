from argparse import Namespace
from typing import Union, Dict, Any

import torch

from src.metrics.metric_base import Logger
from lightning.pytorch.loggers import MLFlowLogger as _MLFlowLogger
from mlflow import MlflowClient
from matplotlib.figure import Figure
from lightning.fabric.loggers.logger import Logger as LoggerBase

class MLFlowLogger(_MLFlowLogger, Logger):

    def log_scalar(self, scalar: torch.Tensor, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_metric(self.run_id,metric_name,scalar.item(),*args,**kwargs)

    def log_figure(self, figure: Figure, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_figure(self.run_id, figure, metric_name,*args,**kwargs)


    def log_dict(self, dict_: dict, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_dict(self.run_id, dict_, metric_name)

    def log_text(self, text: str, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_text(self.run_id, text, metric_name)


class PrintLogger(LoggerBase ,Logger):
    save_dir: str = None
    log_dir: str = None
    @property
    def name(self) -> str:
        return "PrintLogger"

    @property
    def version(self) -> str:
        return "0.0.0"

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        print(f"Hyperparameters: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        print(f"Metrics: {metrics}")


    def log_scalar(self, scalar: torch.Tensor, metric_name: str, *args,**kwargs) -> None:
        print(f"{metric_name}: {scalar.item()}")

    def log_figure(self, figure: Figure, metric_name: str, *args,**kwargs) -> None:
        print(f"{metric_name}: {figure}")

    def log_dict(self, dict_: dict, metric_name: str, *args,**kwarg) -> None:
        print(f"{metric_name}: {dict_}")

    def log_text(self, text: str, metric_name: str, *args,**kwarg) -> None:
        print(f"{metric_name}: {text}")


