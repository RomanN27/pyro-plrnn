import torch

from src.metrics.metric_base import Logger
from lightning.pytorch.loggers import MLFlowLogger as _MLFlowLogger
from mlflow import MlflowClient
from matplotlib.figure import Figure
class MLFlowLogger(_MLFlowLogger, Logger):

    def log_scalar(self, scalar: torch.Tensor, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_metric(self.run_id,metric_name,scalar.item(),*args,**kwargs)

    def log_figure(self, figure: Figure, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_figure(self.run_id, figure, metric_name,*args,**kwargs)


    def log_dict(self, dict_: dict, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_dict(self.run_id, dict_, metric_name)

    def log_text(self, text: str, metric_name: str,*args,**kwargs) -> None:
        self.experiment.log_text(self.run_id, text, metric_name)
