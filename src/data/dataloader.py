import os
from pathlib import Path
import scipy.io
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, Protocol
from lightning import LightningDataModule
from abc import ABC,abstractmethod
from src.data.lorentz.lorentz_system import GeneralLorentzSystem

INDICATE_PATH: Optional[Path] = Path(os.environ.get("INDICATE_PATH")) if os.environ.get("INDICATE_PATH") else None


class TimeIndexer(Protocol):

    def __getitem__(self, item:int|slice)->"TimeSeriesDataset":
        pass


class TimeSeriesDataset(ABC,Dataset):

    @property
    @abstractmethod
    def tloc(self)->TimeIndexer:
        pass


class OneSubjectVanillaLorentzDataSet(TimeSeriesDataset):

    def __init__(self):
        self.tensors = torch.tensor(GeneralLorentzSystem().run_system()).unsqueeze(0)
        self.tloc_ = type('MyClass', (object,), {'__getitem__':lambda self2,item :self.tensors[0,item]})()

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.tensors[item]

    @property
    def tloc(self) ->TimeIndexer:
        return self.tloc_


def create_fake_mrt_data(n_rois, T, n):
    # sample different timeseries length with max length T
    seq_lengths = [T for _ in range(n)]
    # generate complicated time series using a neural network
    data = [torch.randn(t, n_rois) for t in seq_lengths]
    rnn = torch.nn.RNN(n_rois, n_rois)
    with torch.no_grad():
        for i in range(n):
            data[i] = rnn(data[i].unsqueeze(1))[0].squeeze(1)
    return data, data, data

def get_fake_categorical_data(mrt_data:list[torch.Tensor],n_categories: int,n_samples:int):
    lengths = [len(d) for d in mrt_data]
    fake_categorical_data = []
    for i in range(len(mrt_data)):
        randint = torch.randint(0, n_categories, (lengths[i], n_samples))
        fake_categorical_data.append(randint)


    return fake_categorical_data

def get_fake_count_data(mrt_data:list[torch.Tensor],intensity:float,n_samples):

    fake_count_data = [torch.poisson(torch.ones(len(ten), n_samples)*intensity) for ten in mrt_data]

    return fake_count_data





def get_indicate_data() -> torch.Tensor:
    path = INDICATE_PATH
    sub_tensors: list[torch.Tensor] = []
    for file_path in path.glob("**/sub*"):
        mat = scipy.io.loadmat(file_path)
        data = torch.tensor(mat["data"])
        sub_tensors.append(data)

        #get_columns = lambda name, x : [f"{name}_{i}" for i in range(x.shape[1])]
    tensor_sizes = Counter([t.shape for t in sub_tensors])
    most_common_tensor_size = tensor_sizes.most_common(1)[0][0]
    sub_tensors = list(filter(lambda x: x.shape == most_common_tensor_size, sub_tensors))
    tensor = torch.stack(sub_tensors)
    return tensor

class MultiModalDataSet(Dataset):
    def __init__(self, tensors: list[torch.Tensor]):
        self.tensors = tensors

    def n_modalities(self):
        return len(self.tensors[0])
    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]

class IndicateDataSet(Dataset):
    #TODO Use Jagged Tensors instead of list of tensors
    @classmethod
    def from_path(cls, path: Path|str = INDICATE_PATH):
        tensors = cls.get_indicate_data_tensors(path)
        return cls(tensors)

    def __init__(self, tensors: list[torch.Tensor] | torch.Tensor):
        self.tensors = [(tensor - tensor.mean()) / tensor.std() for tensor in tensors]
        self.tloc = tLoc(self)

    @staticmethod
    def get_indicate_data_tensors(indicate_data_path):
        sub_tensors: list[torch.Tensor] = []
        for file_path in indicate_data_path.glob("**/sub*"):
            mat = scipy.io.loadmat(file_path)
            data = torch.tensor(mat["data"], dtype=torch.float32)
            sub_tensors.append(data)

        return sub_tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]


class IndicateDataSetRNN(IndicateDataSet):
    @staticmethod
    def get_mini_batch_mask(mini_batch, seq_lengths):
        mask = torch.zeros(mini_batch.shape[0:2])
        for b in range(mini_batch.shape[0]):
            mask[b, 0: seq_lengths[b]] = torch.ones(seq_lengths[b])
        return mask

    @staticmethod
    def collate_fn(data: list[torch.Tensor]):
        data.sort(key=len, reverse=True)
        reversed_data = [x.flip(0) for x in data]
        seq_lengths = [len(x) for x in reversed_data]
        padded_sequence = pad_sequence(data, batch_first=True)
        padded_reversed_sequence = pad_sequence(reversed_data, batch_first=True)
        packed_reversed_sequence = pack_padded_sequence(padded_reversed_sequence, seq_lengths, batch_first=True)
        batch_mask = IndicateDataLoader.get_mini_batch_mask(padded_sequence, seq_lengths)

        return padded_sequence, packed_reversed_sequence, batch_mask, torch.tensor(seq_lengths)


class tLocIndexer:
    def __init__(self, data_set: IndicateDataSet):
        self.data_set = data_set

    def __getitem__(self, item):
        return IndicateDataSet([tensor[item] for tensor in self.data_set.tensors])


def ts_train_test_split(data: IndicateDataSet, n_test_time_steps: int | list[int]) -> list[IndicateDataSet]:
    #example
    #n_test_time_steps = [10, 20 ,20]
    # data[:-50] , data[:-40] , data[:-20], data

    datasets = [data.tloc[:-t] for t in np.cumsum(n_test_time_steps)[::-1]]
    datasets.append(data)
    return datasets


class IndicateDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=self.collate_fn, *args, **kwargs)

    @staticmethod
    def collate_fn(data: list[torch.Tensor]):
        return data


class IndicateDataModule(LightningDataModule):
    def __init__(self, batch_size: Optional[int] = None, subject_index: Optional[bool] = None, test_steps: int = 20,
                 val_steps: int = 20, data_dir: str = INDICATE_PATH):
        super().__init__()
        assert (batch_size is None) != (subject_index is None)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.subject_index = subject_index
        self.test_steps = test_steps
        self.val_steps = val_steps

    def setup(self, stage: str) -> None:
        if self.subject_index is not None:
            self.train, self.test, self.val = get_data_loaders_of_one_subject(self.subject_index, self.test_steps,
                                                                              self.val_steps)
        else:
            indicate_data = IndicateDataSet.from_path(self.data_dir)
            data_sets = ts_train_test_split(indicate_data,[self.test_steps, self.val_steps])
            self.train, self.test, self.val = [IndicateDataLoader(dataset=ds,batch_size=self.batch_size) for ds in data_sets]


    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test


def get_data():
    path = INDICATE_PATH
    indicate_data = IndicateDataSet.from_path(path)
    train, test, val = random_split(indicate_data, [0.8, 0.1, 0.1])
    train = IndicateDataLoader(train, batch_size=12, shuffle=True)
    test = IndicateDataLoader(test, batch_size=len(test), shuffle=True)
    val = IndicateDataLoader(val, batch_size=len(val), shuffle=True)

    return train, test, val


def get_data_loaders_of_one_subject(subject_index: int, test_steps: int = 20, val_steps: int = 20) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    test, train, val = get_data_sets_of_one_subject(subject_index, test_steps, val_steps)

    train = IndicateDataLoader(train, batch_size=1, shuffle=True)
    test = IndicateDataLoader(test, batch_size=1, shuffle=True)
    val = IndicateDataLoader(val, batch_size=1, shuffle=True)

    return train, test, val


def get_data_sets_of_one_subject(subject_index: int, test_steps: int, val_steps: int) -> Tuple[
    Dataset, Dataset, Dataset]:
    path = INDICATE_PATH
    indicate_data = IndicateDataSet.from_path(path)
    indicate_data_set = IndicateDataSet([indicate_data[subject_index]])
    train, test, val = ts_train_test_split(indicate_data_set, [test_steps, val_steps])
    return test, train, val


def get_fake_data_loader(n_rois=5, T=1000, n=1):
    data, _, _ = create_fake_mrt_data(n_rois, T, n)
    dataset = IndicateDataSet(data)
    dataloader = IndicateDataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader, dataloader, dataloader

def get_fake_multimodal_data_loader(n_rois=5, T=1000, n=1, n_categories=5, n_poisson=1, n_categorical=1):
    data, _, _ = create_fake_mrt_data(n_rois, T, n)
    categorical_data = get_fake_categorical_data(data, n_categories, n_categorical)
    count_data = get_fake_count_data(data, 10, n_poisson)
    multi_modal_data = [torch.cat([data, cat, count],-1) for data, cat, count in zip(data, categorical_data, count_data)]
    dataset = MultiModalDataSet(multi_modal_data)
    dataloader = IndicateDataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader, dataloader, dataloader

class FakeMultiModalDataModule(LightningDataModule):

        def __init__(self, n_rois=5, T=1000, n=1, n_categories=5, n_categorical=1, n_poisson=1):
            super().__init__()
            self.n_rois = n_rois
            self.T = T
            self.n = n
            self.n_categories = n_categories
            self.n_categorical = n_categorical
            self.n_poisson = n_poisson

        def setup(self, stage: str):
            self.train, self.test, self.val = get_fake_multimodal_data_loader(self.n_rois, self.T, self.n, self.n_categories, self.n_poisson,self.n_categorical)

        def train_dataloader(self):
            return self.train

        def val_dataloader(self):
            return self.val

        def test_dataloader(self):
            return self.test
class FakeDataModule(LightningDataModule):

        def __init__(self, n_rois=5, T=1000, n=1):
            super().__init__()
            self.n_rois = n_rois
            self.T = T
            self.n = n

        def setup(self, stage: str):
            self.train, self.test, self.val = get_fake_data_loader(self.n_rois, self.T, self.n)

        def train_dataloader(self):
            return self.train

        def val_dataloader(self):
            return self.val

        def test_dataloader(self):
            return self.test
class FakeDataSet(Dataset):

    def __init__(self, n_rois, T, n):
        self.data = create_fake_mrt_data(n_rois, T, n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == "__main__":
    train, test, val = get_data_loaders_of_one_subject(1)
    train, test, val = get_data()
