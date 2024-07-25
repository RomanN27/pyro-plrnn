import numpy as np
import scipy
import torch

from src.data.time_series_dataset import TimeSeriesDataset, DataType


def load_matlab_data(file_path):
    mat = scipy.io.loadmat(file_path)
    data = torch.tensor(mat["data_loader"], dtype=torch.float32)
    return data


def ts_train_test_split(data: TimeSeriesDataset[DataType], n_test_time_steps: int | list[int]) -> list[
    TimeSeriesDataset[[DataType]]]:
    #example
    #n_test_time_steps = [10, 20 ,20]
    # data_loader[:-50] , data_loader[:-40] , data_loader[:-20], data_loader

    datasets: list[TimeSeriesDataset[[DataType]]] = [data.tloc[:-t] for t in np.cumsum(n_test_time_steps)[::-1]]
    datasets.append(data)
    return datasets