from typing import TypedDict, Unpack, TYPE_CHECKING, Any, Optional, Protocol

import torch
from torchmetrics import Metric as _Metric, MetricCollection as _MetricCollection
from abc import abstractmethod, ABC, ABCMeta
import inspect
from enum import StrEnum
from typing import List, Dict
from matplotlib.pyplot import Figure
from pathlib import Path
from omegaconf import DictConfig
from src.utils.logging_utils import get_plotly_html_string

if TYPE_CHECKING:
    from src.models.forecaster import Forecaster
    from src.models.hidden_markov_model import HiddenMarkovModel
    from src.models.model_sampler import ModelBasedSampler
    from torch import Tensor


class MetricKwarg(TypedDict, total=False):
    batch: "Tensor"
    hmm: "HiddenMarkovModel"
    guide: "ModelBasedSampler"
    forecaster: "Forecaster"



class MetricLogType(StrEnum):
    scalar = "scalar"
    dict = "dict"
    png_figure = "png_figure"
    plotly_figure = "plotly_figure"



class MetricMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        if 'update' in dct:
            kwargs = list(inspect.signature(dct['update']).parameters.values())[1:]
            allowed_kwargs = MetricKwarg.__annotations__.keys()
            for kwarg in kwargs:
                if kwarg.name not in allowed_kwargs:
                    raise ValueError(
                        f"Invalid kwarg {kwarg.name} in {name}.update \n the only allowed kwargs are {allowed_kwargs}"
                        f"see MetricKwarg for the allowed kwargs and their types. the path to the metrickwargs is {MetricKwarg.__module__}")

        super().__init__(name, bases, dct)


class Logger(Protocol):
    def log_scalar(self, scalar: torch.Tensor, metric_name: str, *args,**kwargs) -> None:
        pass

    def log_dict(self, dict_: dict, metric_name: str, *args,**kwarg) -> None:
        pass

    def log_figure(self, figure: Figure, metric_name: str, *args,**kwarg) -> None:
        pass

    def log_text(self, text: str, metric_name: str, *args,**kwarg) -> None:
        pass


class Metric(_Metric, ABC, metaclass=MetricMeta):
    @property
    @abstractmethod
    def log_types(self) -> List[MetricLogType]:
        pass

    def log(self, logger: Logger, _step:str = "",*args,**kwargs):

        name = repr(self)

        for log_type in self.log_types:
            self.log_by_type(log_type, logger, name, _step = _step,*args, **kwargs)

    def log_by_type(self, log_type: MetricLogType, logger: Logger, name: str,  _step: str, *args, **kwargs):
        match log_type:

            case MetricLogType.scalar:
                logger.log_scalar(self.compute(), name, *args, **kwargs)

            case MetricLogType.png_figure:
                figure_path = f"plots/{_step}/{name}.png"

                fig, ax = self.plot()
                logger.log_figure(fig, figure_path, *args, **kwargs)

            case MetricLogType.plotly_figure:
                html = get_plotly_html_string(self.plot())
                figure_path = f"plots/{_step}/{name}.html"
                logger.log_text(html, figure_path, *args, **kwargs)

            case MetricLogType.dict:
                logger.log_dict(self.compute(), name, *args, **kwargs)


class MetricCollection(_MetricCollection):

    def __init__(self, metrics: Dict[str, Metric]|DictConfig[str,Metric], *args, **kwargs):
        super().__init__(dict(metrics), *args, **kwargs)

    def log(self, logger: Logger,  _step: str = "", *args, **kwargs):
        for metric_name, metric in self.items():
            metric.log(logger, _step = _step, *args, **kwargs)
