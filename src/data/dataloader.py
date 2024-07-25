import torch
from torch.utils.data import  DataLoader

from typing import Optional, Callable
from lightning import LightningDataModule

from src.data.time_series_dataset import DataType, TimeSeriesDataset
from src.utils.data_utils import ts_train_test_split


class TimeSeriesDataModule(LightningDataModule):
    def __init__(self, dataset: TimeSeriesDataset[DataType], batch_size: int  = 1,
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn,shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn,shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn,shuffle=True)
