import scipy.io
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable
from lightning import LightningDataModule

from src.data.time_series_dataset import T_el, DataType, TimeSeriesDataset, VanillaTensorTimeSeriesDataset, \
    VanillaListTimeSeriesDataset
from src.data.utils import load_matlab_data

INDICATE_PATH: Optional[Path] = Path(os.environ.get("INDICATE_PATH")) if os.environ.get("INDICATE_PATH") else None

class IndicateListDataSet(VanillaListTimeSeriesDataset):
    #TODO Use Jagged Tensors instead of list of tensors
    @classmethod
    def from_path(cls, path: Path | str = INDICATE_PATH):
        tensors = cls.get_variable_length_data_tensors(path)
        tensors = [(tensor - tensor.mean()) / tensor.std() for tensor in tensors]
        return cls(tensors)

    @staticmethod
    def get_variable_length_data_tensors(indicate_data_path):
        sub_tensors: list[torch.Tensor] = []
        for file_path in indicate_data_path.glob("**/sub*"):
            sub_tensor = load_matlab_data(file_path)
            sub_tensors.append(sub_tensor)
        return sub_tensors


class IndicateTensorDataSet(VanillaTensorTimeSeriesDataset):
    @classmethod
    def from_path(cls, path: Path | str = INDICATE_PATH):
        tensors = cls.get_common_length_data_tensors(path)
        tensors = (tensors - tensors.mean(dim=0)) / tensors.std(dim=0)
        return cls(tensors)


    @staticmethod
    def get_common_length_data_tensors(indicate_data_path: Path):

        sub_tensors: list[torch.Tensor] = []
        for file_path in indicate_data_path.glob("**/sub*"):
            data = load_matlab_data(file_path)
            sub_tensors.append(data)

            # get_columns = lambda name, x : [f"{name}_{i}" for i in range(x.shape[1])]
        tensor_sizes = Counter([t.shape for t in sub_tensors])
        most_common_tensor_size = tensor_sizes.most_common(1)[0][0]
        sub_tensors = list(filter(lambda x: x.shape == most_common_tensor_size, sub_tensors))
        tensor = torch.stack(sub_tensors)
        return tensor