import inspect
import logging
from typing import Type, Optional, cast, TYPE_CHECKING
from inspect import Parameter

from torch import nn as nn

if TYPE_CHECKING:
    from src.models.model_wrappers.model_wrapper import ModelWrapper


def create_wrapped_model_class(model_cls: Type[nn.Module], wrapper_cls: Type["ModelWrapper"]) -> Type[nn.Module]:
    model_cls_init_parameters = get_proper_init_parameters(model_cls)
    wrapper_cls_init_parameters = get_proper_init_parameters(wrapper_cls)


    intersect_parameters = [wrapper_cls_init_parameter for wrapper_cls_init_parameter in wrapper_cls_init_parameters
                            if wrapper_cls_init_parameter in model_cls_init_parameters]

    self_parameter = Parameter("self", kind =Parameter.POSITIONAL_OR_KEYWORD )

    parameters = [self_parameter] + model_cls_init_parameters + [param for param in wrapper_cls_init_parameters if param not in intersect_parameters]
    parameters.sort(key=lambda param: param.default is Parameter.empty, reverse=True)

    # TODO Solve this
    warn_about_common_params(model_cls,  wrapper_cls,intersect_parameters)

    # Create a new __init__ method with the combined parameters
    def __init__(self, *args, **kwargs):
        model_init_kwargs = {k.name: kwargs[k.name] for k in model_cls_init_parameters if k.name in kwargs}
        wrapper_init_kwargs = {k.name: kwargs[k.name] for k in wrapper_cls_init_parameters if k.name in kwargs}

        model_cls.__init__(self, *args, **model_init_kwargs)
        wrapper_instance = wrapper_cls(**wrapper_init_kwargs)
        wrapper_instance(self)

    # Create a new signature for the __init__ method
    __init__.__signature__ = inspect.Signature(parameters)

    WrappedModel = type(
        f"Wrapped{wrapper_cls.__name__}{model_cls.__name__}",
        (model_cls,),
        {"__init__": __init__}
    )

    # Merge metadata
    merge_metadata(WrappedModel, model_cls, wrapper_cls)
    return cast(Type[nn.Module], WrappedModel)


def filter_args_kwargs_and_self_params(parameters: list[Parameter]) -> list[Parameter]:
    #assuming self parameter is the the first entry in the list:
    parameters.pop(0)
    NOT_ALLOWED_PARAMKINDS = [Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL]
    return [parameter for parameter in parameters if parameter.kind not in NOT_ALLOWED_PARAMKINDS]


def get_proper_init_parameters(cls: Type) -> list[Parameter]:
    cls_init_signature = inspect.signature(cls.__init__)
    cls_init_parameters = list(cls_init_signature.parameters.values())
    return filter_args_kwargs_and_self_params(cls_init_parameters)


def warn_about_common_params(model_cls, wrapper_cls, intersect_params:list[Parameter]):

    if intersect_params:
        logging.warning(f"The following parameters are present in both the model class '{model_cls.__name__}' and the "
                        f"wrapper class '{wrapper_cls.__name__}':"
                        f" {intersect_params}. Default behavior will assume they both get the same argument values."
                        f"This may not be intended. Consider renaming initialization variables in the class constructur ")


def merge_metadata(WrappedModel, model_cls, wrapper_cls):
    # Combine the class names
    WrappedModel.__qualname__ = wrapper_cls.__qualname__ + model_cls.__qualname__
    WrappedModel.__name__ = wrapper_cls.__name__ + model_cls.__name__

    # Combine docstrings, treating None as an empty string
    WrappedModel.__doc__ = (wrapper_cls.__doc__ or '') + (model_cls.__doc__ or '')
    WrappedModel.__init__.__doc__ = (wrapper_cls.__init__.__doc__ or '') + (model_cls.__init__.__doc__ or '')

    # Merge annotations
    wrapper_cls_init_annotations = getattr(wrapper_cls.__init__, '__annotations__', {})
    WrappedModel.__init__.__annotations__ = wrapper_cls_init_annotations | model_cls.__init__.__annotations__


def model_wrapper_decorator(wrapper_cls: Type["ModelWrapper"]):
    def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
        return create_wrapped_model_class(model_cls, wrapper_cls)

    return decorator
