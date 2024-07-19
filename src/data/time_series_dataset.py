from abc import ABC, abstractmethod
from typing import TypeVar, ParamSpec, Iterable, Generic, Type

import torch
from torch.utils.data import Dataset

T = TypeVar("T", bound="TimeSeriesDataset")
P = ParamSpec("P")
T_el = TypeVar("T_el")
DataType = TypeVar("DataType", bound=Iterable)


class _tLocIndexer(Generic[DataType]):
    def __init__(self, data_set: "TimeSeriesDataset[DataType]", cls: Type["TimeSeriesDataset[DataType]"]):
        self.data_set = data_set
        self.cls = cls

    def __getitem__(self, item: int | slice) -> "TimeSeriesDataset[DataType]":
        return self.cls(self.cls.time_slice(item, self.data_set.tensors))


class TimeSeriesDataset(ABC, Dataset, Generic[DataType[T_el]]):

    def __init__(self, tensors: DataType):
        self.tensors = tensors

    @property
    def tloc(self) -> _tLocIndexer[DataType]:
        return _tLocIndexer[DataType](self, type(self))

    @abstractmethod
    @classmethod
    def time_slice(cls, item: int | slice, tensors: DataType) -> DataType:
        pass

    def __getitem__(self, item):
        return self.tensors[item]

    def __len__(self):
        return len(self.tensors)


class VanillaTensorTimeSeriesDataset(TimeSeriesDataset[torch.Tensor]):

    def time_slice(cls, item: int | slice, tensors: torch.Tensor) -> torch.Tensor:
        #assuming time is in second  dimension
        return tensors[:, item]


class VanillaListTimeSeriesDataset(TimeSeriesDataset[list[torch.Tensor]]):
    def time_slice(cls, item: int | slice, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        #assuming time is in second  dimension
        return [tensor[:, item] for tensor in tensors]
