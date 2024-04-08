from pathlib import Path
import scipy.io
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def get_indicate_data() -> torch.Tensor:
    path = Path(r"C:\Users\roman.nefedov\OneDrive\Desktop\INDICATE")
    sub_tensors:list[torch.Tensor] = []
    for file_path in path.glob("**/sub*"):

        mat = scipy.io.loadmat(file_path)
        data = torch.tensor(mat["data"])
        sub_tensors.append(data)

        #get_columns = lambda name, x : [f"{name}_{i}" for i in range(x.shape[1])]
    tensor_sizes = Counter([t.shape for t in sub_tensors])
    most_common_tensor_size = tensor_sizes.most_common(1)[0][0]
    sub_tensors = list(filter(lambda x: x.shape == most_common_tensor_size,sub_tensors))
    tensor = torch.stack(sub_tensors)
    return tensor


class IndicateDataSet(Dataset):

    def __init__(self,indicate_data_path: Path):
        self.tensors = self.get_indicate_data_tensors(indicate_data_path)
        self.tensors = [(tensor - tensor.mean())/tensor.std() for tensor in self.tensors]


    @staticmethod
    def get_indicate_data_tensors(indicate_data_path):
        sub_tensors: list[torch.Tensor] = []
        for file_path in indicate_data_path.glob("**/sub*"):
            mat = scipy.io.loadmat(file_path)
            data = torch.tensor(mat["data"],dtype=torch.float32)
            sub_tensors.append(data)


        return sub_tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]

class IndicateDataLoader(DataLoader):

    def __init__(self,*args,**kwargs):
        super().__init__(collate_fn=self.collate_fn, *args,**kwargs)

    @staticmethod
    def get_mini_batch_mask(mini_batch, seq_lengths):
        mask = torch.zeros(mini_batch.shape[0:2])
        for b in range(mini_batch.shape[0]):
            mask[b, 0: seq_lengths[b]] = torch.ones(seq_lengths[b])
        return mask
    @staticmethod
    def collate_fn(data: list[torch.Tensor]):
        data.sort(key =len,reverse=True)
        reversed_data = [x.flip(0) for x in data]
        seq_lengths = [len(x) for x in reversed_data]
        padded_sequence = pad_sequence(data,batch_first=True)
        padded_reversed_sequence = pad_sequence(reversed_data,batch_first=True)
        packed_reversed_sequence = pack_padded_sequence(padded_reversed_sequence,seq_lengths,batch_first=True)
        batch_mask = IndicateDataLoader.get_mini_batch_mask(padded_sequence,seq_lengths)

        return padded_sequence, packed_reversed_sequence, batch_mask, torch.tensor(seq_lengths)


def get_data():
    path = Path(r"C:\Users\roman.nefedov\OneDrive\Desktop\INDICATE")
    indicate_data = IndicateDataSet(path)
    train, test, val = random_split(indicate_data, [0.8, 0.1, 0.1])
    train = IndicateDataLoader(train, batch_size=12, shuffle=True)
    test = IndicateDataLoader(test, batch_size=len(test), shuffle=True)
    val = IndicateDataLoader(val, batch_size=len(val), shuffle=True)

    return train, test, val

def get_data_of_one_subject(subject_index: int):
    path = Path(r"C:\Users\roman.nefedov\OneDrive\Desktop\INDICATE")
    indicate_data = IndicateDataSet(path)
    indicate_data.tensors = [indicate_data.tensors[subject_index]]

    train = IndicateDataLoader(indicate_data, batch_size=1, shuffle=True)
    test = IndicateDataLoader(indicate_data, batch_size=1, shuffle=True)
    val = IndicateDataLoader(indicate_data, batch_size=1, shuffle=True)

    return train, test, val


if __name__  == "__main__":
    get_data_of_one_subject(1)