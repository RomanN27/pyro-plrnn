import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from typing import Tuple
def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output


def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0: seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask

def collate_fn(data: list[torch.Tensor]):
    data.sort(key =len,reverse=True)
    reversed_data = [x.flip(0) for x in data]
    seq_lengths = [len(x) for x in reversed_data]
    padded_sequence = pad_sequence(data,batch_first=True)
    padded_reversed_sequence = pad_sequence(reversed_data,batch_first=True)
    packed_reversed_sequence = pack_padded_sequence(padded_reversed_sequence,seq_lengths,batch_first=True)
    batch_mask = get_mini_batch_mask(padded_sequence,seq_lengths)

    return padded_sequence, packed_reversed_sequence, batch_mask, torch.tensor(seq_lengths)

def collate_fn_2(data: list[torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
    padded_sequence, _, batch_mask, _ = collate_fn(data)
    return padded_sequence, batch_mask
