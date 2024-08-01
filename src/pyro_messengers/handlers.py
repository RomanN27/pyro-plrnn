from src.pyro_messengers import ObservedBatchMessenger, SubSpaceReplayMessenger, \
    ForcingIntervalReplayMessenger, SampleMeanMessenger
from typing import Optional, overload, ParamSpec, TypeVar, Callable, Union
from pyro.poutine.handlers import _make_handler
import torch
from pyro.poutine.trace_struct import Trace

_P = ParamSpec("_P")
#_P2 = ParamSpec("_P2")
#IP = ParamSpec("IP") # Instantiation Parameters of Messenger class
_T = TypeVar("_T")
#_T2 = TypeVar("_T2")
#MT = TypeVar("MT",bound=Messenger)




# def check_fn(fn):
#     if fn is not None and not (
#             callable(fn) or isinstance(fn, collections.abc.Iterable)
#     ):
#         raise ValueError(
#             f"{fn} is not callable, did you mean to pass it as a keyword arg?"
#         )
#
# def add_doc_to_handler(func, handler,msngr_cls,module):
#     handler.__doc__ = (
#             """Convenient wrapper of :class:`~pyro.poutine.{}.{}` \n\n""".format(
#                 func.__name__ + "_messenger", msngr_cls.__name__
#             )
#             + (msngr_cls.__doc__ if msngr_cls.__doc__ else "")
#     )
#     if module is not None:
#         handler.__module__ = module
# def _make_handler(msngr_cls:Callable[IP,MT], module=None)->Callable[[Callable[_P, _T]],Callable[[Optional[Callable[_P2,_T2]],_P],MT|Callable[]]]:
#     def handler_decorator(func:Callable[_P, _T]):
#         @functools.wraps(func)
#         def handler(fn:Optional[Callable[_P2,_T2]]=None, *args:IP.args, **kwargs:IP.kwargs)->MT|:
#             check_fn(fn)
#             msngr_instance = msngr_cls(*args, **kwargs)
#
#             if fn is None:
#                 return msngr_instance
#
#             return functools.update_wrapper(msngr_instance(fn), fn, updated=())
#
#         add_doc_to_handler(func, handler,msngr_cls,module)
#         return handler
#
#
#
#
#     return handler_decorator



@overload
def subspace_replay(
        subspace_dim: int,
        group_name: str,
        trace: "Trace"
) -> SubSpaceReplayMessenger: ...


@overload
def subspace_replay(
        fn: Callable[_P, _T],
        subspace_dim: int,
        group_name: str,
        trace: "Trace",
) -> Callable[_P, _T]: ...


@_make_handler(SubSpaceReplayMessenger)
def subspace_replay(  # type: ignore[empty-body]
        fn: Optional[Callable[_P, _T]],
        group_name: str,
        subspace_dim: int,
        trace: "Trace"
) -> Union[SubSpaceReplayMessenger, Callable[_P, _T]]: ...


@overload
def observe(
        batch: torch.Tensor, observation_group_symbol: str
) -> ObservedBatchMessenger: ...


@overload
def observe(
        fn: Callable[_P, _T],
        batch: torch.Tensor, observation_group_symbol: str
) -> Callable[_P, _T]: ...


@_make_handler(ObservedBatchMessenger)
def observe(fn: Callable[_P, _T],
            batch: torch.Tensor, observation_group_symbol: str) -> Union[Callable[_P, _T], ObservedBatchMessenger]: ...


@overload
def force(
        trace: Trace,
        forcing_interval: int,
        latent_group_name: str,
        alpha: float = 1.,
        subspace_dim: Optional[int] = None
) -> ForcingIntervalReplayMessenger: ...


@overload
def force(
        fn: Callable[_P, _T],
        trace: Trace,
        forcing_interval: int,
        latent_group_name: str,
        alpha: float = 1.,
        subspace_dim: Optional[int] = None
) -> Callable[_P, _T]: ...


@_make_handler(ForcingIntervalReplayMessenger)
def force(fn: Callable[_P, _T],
          trace: Trace,
          forcing_interval: int,
          latent_group_name: str,
          alpha: float = 1.,
          subspace_dim: Optional[int] = None) -> Union[Callable[_P, _T], ForcingIntervalReplayMessenger]:
    ...


@overload
def mean(fn: None) ->  ForcingIntervalReplayMessenger:...

@overload
def mean(fn: Callable[_P, _T]) ->  Callable[_P, _T]:...


@_make_handler(SampleMeanMessenger)
def mean(fn: Optional[Callable[_P, _T]]) -> Union[Callable[_P, _T], ForcingIntervalReplayMessenger]:...