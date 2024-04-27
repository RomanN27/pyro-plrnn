from pyro.distributions import TorchDistribution, Normal, LogNormal, Poisson, Multinomial, Independent
from typing import Type
from torch import Tensor
import torch
from src.utils.custom_typehint import TensorIterable
import numpy as np
from typing import Callable
import functools



class ProductDistribution(TorchDistribution):
    distributions: list[Type[TorchDistribution]] = []
    def __init__(self, *distribution_parameters:list[Tensor]):

        self.distribution_instances: list[TorchDistribution] = [d(*dp) for d,dp in zip(self.distributions,distribution_parameters)]
        batch_dims = [d.batch_shape for d in self.distribution_instances]
        assert len(set(batch_dims)) == 1
        batch_dim = batch_dims[0]

        event_dims = [d.event_shape for d in self.distribution_instances]
        lengths = [len(e) for e in event_dims]
        assert (m:=max(lengths)) - min(lengths) <= 1
        self.distribution_indices_with_shorter_event_dims = [i for i,d in enumerate(self.distribution_instances) if len(d.event_shape) != m]


    @property
    def event_dim(self) -> int:
        return len(max(self.event_shape, key=len))

    def event_shape(self) -> list[torch.Size]:
        return [dist.event_shape for dist in self.distribution_instances]

    def mask(self, mask) -> "ProductDistribution":
        self.distribution_instances = [dist.mask(mask) for dist in self.distribution_instances]
        return self


    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample_list = [d.sample(sample_shape) for d in self.distribution_instances]
        sample_list = [s.unsqueeze(-1) if i in self.distribution_indices_with_shorter_event_dims else  s for i,s in enumerate(sample_list) ]
        return torch.cat(sample_list,-1)

    def log_prob(self, value: TensorIterable) -> torch.Tensor:
        dist_dims = np.cumsum([i[0] if i else 1 for i in self.event_shape])
        values = [value[...,i:j] for i,j in zip([0,*dist_dims], dist_dims)]
        values = [value.squeeze(-1) if i in self.distribution_indices_with_shorter_event_dims else value for i,value in enumerate(values)]
        log_probs = [d.log_prob(v) for v, d in zip(values, self.distribution_instances)]
        log_prob = sum(log_probs)
        return log_prob

import functools


def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class Multinomial1(Multinomial):

    def __init__(self,*args, total_count = 1, **kwargs):
        super().__init__(total_count, *args,**kwargs)


def get_product_distribution(sub_distributions: dict[str: Type[TorchDistribution]]) -> Type[ProductDistribution]:
    sub_distributions = list(sub_distributions.values())
    mmd = type(" x ".join([d.__name__ for d in sub_distributions]), (ProductDistribution,), {"distributions": sub_distributions})
    return mmd

if __name__ == "__main__":
    distributions_c = [Normal, Multinomial, Poisson]

    mmd = get_product_distribution(*distributions_c)
    print(mmd)

    n = 10

    a = torch.randn((n,5))
    b = torch.exp(a)
    p = b.T / b.sum(1)
    p  =p.T
    distributions_parameters = [[torch.randn(10),torch.randn(10)**2],
                                [1,p],
                                [torch.randn(10)**2] ]

    d = mmd(*distributions_parameters)

    sample  =d.sample()

    p = d.log_prob(sample)
    print(sample)

    d.event_shape