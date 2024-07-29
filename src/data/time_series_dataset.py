from abc import ABC, abstractmethod
from typing import TypeVar, ParamSpec, Iterable, Generic, Type

import torch
from torch.utils.data import Dataset

T = TypeVar("T", bound="TimeSeriesDataset")
P = ParamSpec("MuParams")
DataType = TypeVar("DataType", bound=Iterable)


class _tLocIndexer(Generic[DataType]):
    def __init__(self, data_set: "TimeSeriesDataset[DataType]", cls: Type["TimeSeriesDataset[DataType]"]):
        self.data_set = data_set
        self.cls = cls

    def __getitem__(self, item: int | slice) -> "TimeSeriesDataset[DataType]":
        return self.cls(self.cls.time_slice(item, self.data_set.tensors),**self.data_set.kwargs)


class TimeSeriesDataset(ABC, Dataset, Generic[DataType]):

    def __init__(self, tensors: DataType,**kwargs):
        self.tensors = tensors

        self.kwargs = kwargs

    @property
    def tloc(self) -> _tLocIndexer[DataType]:
        return _tLocIndexer[DataType](self, type(self))

    @classmethod
    @abstractmethod
    def time_slice(cls, item: int | slice, tensors: DataType) -> DataType:
        pass

    def __getitem__(self, item):
        return self.tensors[item]

    def __len__(self):
        return len(self.tensors)


class VanillaTensorTimeSeriesDataset(TimeSeriesDataset[torch.Tensor]):
    def __init__(self, tensors: torch.Tensor,**kwargs):
        tensors = (tensors-tensors.mean(-2))/tensors.std(-2)
        super().__init__(tensors,**kwargs)

    @classmethod
    def time_slice(cls, item: int | slice, tensors: torch.Tensor) -> torch.Tensor:
        #assuming time is in second  dimension
        return tensors[:, item]


class VanillaListTimeSeriesDataset(TimeSeriesDataset[list[torch.Tensor]]):
    @classmethod
    def time_slice(cls, item: int | slice, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        #assuming time is in second  dimension
        return [tensor[:, item] for tensor in tensors]


class TimeSeriesChunkDataset(VanillaTensorTimeSeriesDataset):

    def __init__(self, tensors: torch.Tensor, chunk_length: int):
        assert chunk_length < tensors.size(1)
        super().__init__(tensors,chunk_length= chunk_length)
        self.chunk_length = self.kwargs["chunk_length"]

    @property
    def n_chunks(self):
        return self.tensors.size(1) - self.chunk_length + 1

    def __len__(self):
        return self.n_chunks * len(self.tensors)

    def __getitem__(self, item):
        time_series_item = item // self.n_chunks
        chunk_item = item % self.n_chunks

        orig_time_series = super().__getitem__(time_series_item)
        time_series_chunk = orig_time_series[chunk_item: chunk_item + self.chunk_length]

        return time_series_chunk


