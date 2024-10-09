from src.models.cov_mixins import FixedCovarianceMixin, ConstantCovarianceMixin
from src.models.initializer_mixins import (ShallowInitializer, UniformAlphaInitializer, UniformThresholdHInitializer,
                                           NormalizedPositiveDefiniteInitializer, ZeroBiasInitializer)
from src.utils.mixin_utils import get_nested_signature
from src.models.normalizer_mixins import PLRNNNormalizerMixin
from src.models.transition_models.plrnns.raw_plrnns import _DendPLRNN, _ShallowPLRNN, _ClippedDendPLRNN, \
    _ClippedShallowPLRNN
from src.models.hierarchization_mixin import DeterministicHierarchizationMixin
import inspect
from functools import wraps

plrnn_default_mixins = [ZeroBiasInitializer, NormalizedPositiveDefiniteInitializer]
import torch
import inspect
from copy import deepcopy




dend_default_mixins = [UniformAlphaInitializer, UniformThresholdHInitializer]


class DendPLRNN(*plrnn_default_mixins,
                *dend_default_mixins,
                _DendPLRNN): ...


class ClippedDendPLRNN(*plrnn_default_mixins,
                       *dend_default_mixins,
                       _ClippedDendPLRNN): ...


class ShallowPLRNN(*plrnn_default_mixins,
                   ShallowInitializer,
                   _ShallowPLRNN): ...


class ClippedShallowPLRNN(*plrnn_default_mixins,
                          ShallowInitializer,
                          _ClippedShallowPLRNN): ...


class FixedCovDendPLRNN(FixedCovarianceMixin, DendPLRNN): ...


class FixedCovClippedDendPLRNN(FixedCovarianceMixin, ClippedDendPLRNN): ...


class FixedCovShallowPLRNN(FixedCovarianceMixin, ShallowPLRNN): ...


class FixedCovClippedShallowPLRNN(FixedCovarianceMixin, ClippedShallowPLRNN): ...


class ConstantCovDendPLRNN(ConstantCovarianceMixin, DendPLRNN): ...


class ConstantCovClippedDendPLRNN(ConstantCovarianceMixin, ClippedDendPLRNN): ...


class ConstantCovShallowPLRNN(ConstantCovarianceMixin, ShallowPLRNN): ...


class ConstantCovClippedShallowPLRNN(ConstantCovarianceMixin, ClippedShallowPLRNN): ...

class HierarchicalClippedShallowPLRNN(DeterministicHierarchizationMixin, ClippedShallowPLRNN): ...


#TODO fix signature

def build_plrnn_with_mixins(cls, *mixins, name : str = None):

    class NewClass(cls, *mixins):
        @classmethod
        def get_signature(cls):
            return get_nested_signature(cls)

    if name is not None:
        NewClass.__name__ = name
        NewClass.__qualname__ = name


    return NewClass

#we make a decorator out of it

def attach_mixins(*mixins):
    def decorator(cls):
        name = cls.__name__
        return build_plrnn_with_mixins(cls,*mixins,name=name)

    return decorator

