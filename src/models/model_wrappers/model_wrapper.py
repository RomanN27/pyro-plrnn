from typing import TypeVar, Type, Callable
import torch.nn as nn
from abc import ABC, abstractmethod

from src.models.model_wrappers._model_wrapper_utils import create_wrapped_model_class


class ModelWrapper(ABC):
    """
    A base class for wrapping PyTorch models.

    This abstract class serves as a protocol for creating wrappers around PyTorch `nn.Module` models.
    Wrappers can be used to add additional functionality or modify behavior without altering the original
    model's code.

    Methods:
        __call__(model: nn.Module) -> None:
            Abstract method that must be implemented by subclasses. This method defines how the wrapper
            interacts with the model.

        get_decorator() -> Callable[[Type[nn.Module]], Type[nn.Module]]:
            Class method that returns a decorator. This decorator can be used to wrap a PyTorch model class
            with the current wrapper.

        wrap(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            Class method that wraps a given PyTorch model class with the current wrapper and returns the
            wrapped model class.

    Usage:
        Subclass this `ModelWrapper` and implement the `__call__` method to define custom wrapping behavior.
        Use the `wrap` method or the `get_decorator` method to apply the wrapper to a PyTorch model class.

    Examples:
        Example 1: Initializing wrapper with an attribute
        -------------------------------------------------
        This example demonstrates how to use a wrapper to add an attribute to the model and initialize the wrapper.

        ```python
        class AddAttributeWrapper(ModelWrapper):
            def __init__(self, attribute_value: str):
                self.attribute_value = attribute_value

            def __call__(self, model: nn.Module) -> None:
                model.new_attribute = self.attribute_value

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

        WrappedModel = AddAttributeWrapper.wrap(MyModel)
        model = WrappedModel(attribute_value="example")
        print(model.new_attribute)  # Output: example
        ```

        Example 2: Initializing wrapper with a forward hook
        ---------------------------------------------------
        This example shows how to use a wrapper to add a forward hook to a model, allowing you to
        execute custom code each time the model's `forward` method is called, and initialize the wrapper with parameters.

        ```python
        class ForwardHookWrapper(ModelWrapper):
            def __init__(self, hook_message: str):
                self.hook_message = hook_message

            def __call__(self, model: nn.Module) -> None:
                def hook(module, input, output):
                    print(self.hook_message)
                model.register_forward_hook(hook)

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        WrappedModel = ForwardHookWrapper.wrap(MyModel)
        model = WrappedModel(hook_message="Forward hook called")
        x = torch.randn(1, 10)
        model(x)  # Output: Forward hook called
        ```

        Example 3: Initializing wrapper to modify the model's forward method
        --------------------------------------------------------------------
        This advanced example illustrates how to modify the model's `forward` method using a wrapper and initialize the wrapper with parameters.

        ```python
        class ModifyForwardWrapper(ModelWrapper):
            def __init__(self, message: str):
                self.message = message

            def __call__(self, model: nn.Module) -> None:
                original_forward = model.forward

                def new_forward(*args, **kwargs):
                    print(self.message)
                    return original_forward(*args, **kwargs)

                model.forward = new_forward

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        WrappedModel = ModifyForwardWrapper.wrap(MyModel)
        model = WrappedModel(message="Modified forward method called")
        x = torch.randn(1, 10)
        model(x)  # Output: Modified forward method called
        ```
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, model: nn.Module) -> None:
        """
        Defines the behavior of the wrapper when applied to a model.

        Args:
            model (nn.Module): The PyTorch model to be wrapped.
        """
        pass

    @classmethod
    def get_decorator(cls) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        """
        Returns a decorator that can be used to wrap a PyTorch model class with the current wrapper.

        Returns:
            Callable[[Type[nn.Module]], Type[nn.Module]]: A decorator function.
        """

        def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            return create_wrapped_model_class(model_cls, cls)

        return decorator

    @classmethod
    def wrap(cls, model_cls: Type[nn.Module]) -> Type[nn.Module]:
        """
        Wraps a given PyTorch model class with the current wrapper and returns the wrapped model class.

        Args:
            model_cls (Type[nn.Module]): The PyTorch model class to be wrapped.

        Returns:
            Type[nn.Module]: The wrapped model class.
        """
        return create_wrapped_model_class(model_cls, cls)
