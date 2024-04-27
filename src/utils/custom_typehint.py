from typing import Protocol
import torch

from typing import TypeAlias, Iterable

TensorIterable: TypeAlias = Iterable[torch.Tensor]