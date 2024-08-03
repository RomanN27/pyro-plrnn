import torch
from torch.utils.data import DataLoader as DataLoader

from typing import Optional, Callable, Generic, TypeVar
from lightning import LightningDataModule

from src.data.time_series_dataset import DataType, TimeSeriesDataset, TimeSeriesChunkDataset, VanillaTensorTimeSeriesDataset
from src.utils.data_utils import ts_train_test_split

DatasetType = TypeVar("DatasetType", bound=TimeSeriesDataset)




class TimeSeriesDataModule(LightningDataModule, Generic[DatasetType, DataType]):
    def __init__(self, dataset: DatasetType, batch_size: int = 1,
                 test_steps: int = 20,
                 val_steps: int = 20, collate_fn: Optional[Callable[[list], torch.Tensor]] = None):
        super().__init__()

        self.batch_size = batch_size
        self.test_steps = test_steps
        self.val_steps = val_steps
        self.dataset = dataset
        self.collate_fn = collate_fn

    def setup(self, stage: str) -> None:
        self.train, self.test, self.val = ts_train_test_split(self.dataset, [self.test_steps, self.val_steps])

    def train_dataloader(self) -> DataLoader[ DataType]:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self) -> DataLoader[DataType]:
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def test_dataloader(self) -> DataLoader[DataType]:
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False)


class ChunkTimeSeriesDataModule(TimeSeriesDataModule[TimeSeriesChunkDataset, DataType]):

    def __init__(self, train_batch_size = 32,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_batch_size = train_batch_size

    def train_dataloader(self) -> DataLoader[ DataType]:
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self) -> DataLoader[DataType]:
        unwrapped_dataset = VanillaTensorTimeSeriesDataset(self.val.tensors)
        return DataLoader(unwrapped_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False)
    def test_dataloader(self) -> DataLoader[DataType]:
        unwrapped_dataset = VanillaTensorTimeSeriesDataset(self.dataset.tensors)
        return DataLoader(unwrapped_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False)
