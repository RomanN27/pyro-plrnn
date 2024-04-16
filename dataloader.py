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
from typing import Optional

INDICATE_PATH: Optional[Path] = Path(os.environ.get("INDICATE_PATH")) if os.environ.get("INDICATE_PATH") else None


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


class IndicateDataSet(Dataset):

    @classmethod
    def from_path(cls, path: Path = INDICATE_PATH):
        tensors = cls.get_indicate_data_tensors(path)
        return cls(tensors)

    def __init__(self, tensors: list[torch.Tensor] | torch.Tensor):
        self.tensors = [(tensor - tensor.mean()) / tensor.std() for tensor in tensors]
        self.tloc = tLocIndexer(self)

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


def get_data():
    path = INDICATE_PATH
    indicate_data = IndicateDataSet.from_path(path)
    train, test, val = random_split(indicate_data, [0.8, 0.1, 0.1])
    train = IndicateDataLoader(train, batch_size=12, shuffle=True)
    test = IndicateDataLoader(test, batch_size=len(test), shuffle=True)
    val = IndicateDataLoader(val, batch_size=len(val), shuffle=True)

    return train, test, val


def get_data_of_one_subject(subject_index: int, test_steps: int = 20, val_steps: int = 20):
    path = INDICATE_PATH
    indicate_data = IndicateDataSet.from_path(path)
    indicate_data_set = IndicateDataSet([indicate_data[subject_index]])

    train, test, val = ts_train_test_split(indicate_data_set, [test_steps, val_steps])

    train = IndicateDataLoader(train, batch_size=1, shuffle=True)
    test = IndicateDataLoader(test, batch_size=1, shuffle=True)
    val = IndicateDataLoader(val, batch_size=1, shuffle=True)

    return train, test, val


def get_fake_data_loader(n_rois=5, T=1000, n=1):
    data, _, _ = create_fake_mrt_data(n_rois, T, n)
    dataset = IndicateDataSet(data)
    dataloader = IndicateDataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader, dataloader, dataloader


class FakeDataSet(Dataset):

    def __init__(self, n_rois, T, n):
        self.data = create_fake_mrt_data(n_rois, T, n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == "__main__":
    train, test, val = get_data_of_one_subject(1)
    train, test, val = get_data()
