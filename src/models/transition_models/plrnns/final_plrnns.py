from src.models.cov_mixins import FixedCovarianceMixin, ConstantCovarianceMixin
from src.models.initializer_mixins import (ShallowInitializer, UniformAlphaInitializer, UniformThresholdHInitializer,
                                           NormalizedPositiveDefiniteInitializer, ZeroBiasInitializer)

from src.models.normalizer_mixins import PLRNNNormalizerMixin
from src.models.transition_models.plrnns.raw_plrnns import _DendPLRNN, _ShallowPLRNN, _ClippedDendPLRNN, \
    _ClippedShallowPLRNN

import inspect
from functools import wraps

plrnn_default_mixins = [ZeroBiasInitializer, NormalizedPositiveDefiniteInitializer, PLRNNNormalizerMixin]
import torch
import inspect


class CombineInitMeta(type):
    def __new__(cls, name, bases, class_dict):
        init_params = []
        for base in bases:
            if hasattr(base, '__init__'):
                sig = inspect.signature(base.__init__)
                for param in sig.parameters.values():
                    if param.name not in {'self', 'args', 'kwargs'}:
                        init_params.append(param)

        # Remove duplicates while preserving order
        seen = set()
        init_params = [p for p in init_params if p.name not in seen and not seen.add(p.name)]

        # Create a new __init__ with combined parameters
        init_sig = inspect.Signature(init_params)

        new_cls = type.__new__(cls, name, bases, class_dict)
        new_cls.__init__.__signature__ = init_sig

        return new_cls


dend_default_mixins = [UniformAlphaInitializer, UniformThresholdHInitializer]


class DendPLRNN(*plrnn_default_mixins,
                *dend_default_mixins,
                _DendPLRNN,
                metaclass=CombineInitMeta): ...


class ClippedDendPLRNN(*plrnn_default_mixins,
                       *dend_default_mixins,
                       _ClippedDendPLRNN,
                       metaclass=CombineInitMeta): ...


class ShallowPLRNN(*plrnn_default_mixins,
                   ShallowInitializer,
                   _ShallowPLRNN,
                   metaclass=CombineInitMeta): ...


class ClippedShallowPLRNN(*plrnn_default_mixins,
                          ShallowInitializer,
                          _ClippedShallowPLRNN,
                          metaclass=CombineInitMeta): ...


class FixedCovDendPLRNN(FixedCovarianceMixin, DendPLRNN, metaclass=CombineInitMeta): ...


class FixedCovClippedDendPLRNN(FixedCovarianceMixin, ClippedDendPLRNN, metaclass=CombineInitMeta): ...


class FixedCovShallowPLRNN(FixedCovarianceMixin, ShallowPLRNN, metaclass=CombineInitMeta): ...


class FixedCovClippedShallowPLRNN(FixedCovarianceMixin, ClippedShallowPLRNN, metaclass=CombineInitMeta): ...


class ConstantCovDendPLRNN(ConstantCovarianceMixin, DendPLRNN, metaclass=CombineInitMeta): ...


class ConstantCovClippedDendPLRNN(ConstantCovarianceMixin, ClippedDendPLRNN, metaclass=CombineInitMeta): ...


class ConstantCovShallowPLRNN(ConstantCovarianceMixin, ShallowPLRNN, metaclass=CombineInitMeta): ...


class ConstantCovClippedShallowPLRNN(ConstantCovarianceMixin, ClippedShallowPLRNN, metaclass=CombineInitMeta): ...

#TODO fix signature
