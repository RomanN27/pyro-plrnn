import numpy as np
from lightning import LightningDataModule
from typing import Callable
from src.data.dataloader import IndicateDataSet, IndicateDataLoader
import torch
torch.sin

class FakeFunctionalData(LightningDataModule):

    def __init__(self, fun: Callable, T: float, ds: float):
        super().__init__()
        self.train_ds = IndicateDataSet([fun(torch.arange(0,T,ds)).reshape(-1,1)])
        self.test_ds = IndicateDataSet([fun(torch.arange(T,T+0.1*T,ds)).reshape(-1,1)])
        self.val_ds = IndicateDataSet([fun(torch.arange(T+0.1,T+0.2,ds)).reshape(-1,1)])


    def setup(self, stage: str):
        self.train, self.test, self.val = (
            IndicateDataLoader(self.train_ds),IndicateDataLoader(self.test_ds),IndicateDataLoader(self.val_ds))

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test


