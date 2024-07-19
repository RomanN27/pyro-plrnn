import torch
from torch.utils.data import  DataLoader

from typing import Optional, Callable
from lightning import LightningDataModule

from src.data.time_series_dataset import T_el, DataType, TimeSeriesDataset
from src.data.utils import ts_train_test_split


class TimeSeriesDataModule(LightningDataModule):
    def __init__(self, dataset: TimeSeriesDataset[DataType[T_el]], batch_size: Optional[int] = None,
                 test_steps: int = 20,
                 val_steps: int = 20, collate_fn: Optional[Callable[[list[T_el]], torch.Tensor]] = None):
        super().__init__()

        self.batch_size = batch_size
        self.test_steps = test_steps
        self.val_steps = val_steps
        self.dataset = dataset
        self.collate_fn = collate_fn

    def setup(self, stage: str) -> None:
        self.train, self.test, self.val = ts_train_test_split(self.dataset, [self.test_steps, self.val_steps])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn)
