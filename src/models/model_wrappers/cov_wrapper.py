from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Type, cast

from torch import nn as nn

from src.models.model_wrappers.model_wrapper import ModelWrapper


class MyMeta(ABCMeta):
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)

        if getattr(obj, "cov_module") is None:
            raise ValueError("You have to instantiate a Covariance Module in your Constructor")
        return obj


class CovarianceWrapper(ModelWrapper, metaclass=MyMeta):
    cov_module: nn.Module = None

    def __call__(self, model: nn.Module) -> None:
        old_forward = model.forward
        model.forward = self.get_cov_extended_forward(old_forward)

    def get_cov_extended_forward(self, old_forward):
        def cov_extended_forward(z):  # I am not understanding why this one doesnt need a self like first argument
            return old_forward(z), self.cov_module(z)

        return cov_extended_forward


def create_covariance_wrapper(cov_module_cls: Type[nn.Module]) -> CovarianceWrapper:
    """
    Function to create a CovarianceWrapper with a given covariance module class.

    Args:
        cov_module_cls: The class of the covariance module.

    Returns:
        A decorator function to create a CovarianceWrapper class with the specified covariance module class.
    """

    @wraps(cov_module_cls.__init__)
    def __init__(self, *args, **kwargs):
        self.cov_module = cov_module_cls(*args, **kwargs)

    CovarianceWrapperChild = type(
        f"{cov_module_cls.__name__}Wrapper",
        (CovarianceWrapper,),
        {"__init__": __init__}
    )

    return cast(CovarianceWrapper, CovarianceWrapperChild)
