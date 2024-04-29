from pyro.distributions import TorchDistribution, Normal, LogNormal, Poisson, Multinomial, Independent
from typing import Type
from torch import Tensor
import torch
from src.utils.custom_typehint import TensorIterable
import numpy as np
from typing import Callable
import functools
import src.utils.custom_independent
class ProductEventShape:
    # Not functional
    # Would have been used to implement a product distribution over distributions with different event shape dimensions
    def __init__(self, *event_shapes: torch.Size):
        self.event_shapes = event_shapes
    def __radd__(self, other: torch.Size) -> "ProductBatchAndEventShape":
        return ProductBatchAndEventShape(other, self)

    def __iter__(self):
        return iter(self.event_shapes)

class ProductBatchAndEventShape:
    #Not functional
    # Would have been used to implement a product distribution over distributions with different event shape dimensions
    def __init__(self,batch_shape: torch.Size, event_shape: ProductEventShape):
        self.batch_shape = batch_shape
        self.event_shape = event_shape

    @property
    def blen(self):
        return len(self.batch_shape)

    def __getitem__(self, item: int | slice):

        if isinstance(item, int):
            if item < self.blen:
                return self.batch_shape[item]


        elif isinstance(item,slice):
            if item.stop and item.stop <= self.blen:
                return self.batch_shape[item]
            else:

                return ProductEventShape(*[self.batch_shape[item] + es for es in self.event_shape])








class GeneralProductDistribution(TorchDistribution):
    #not functional

    # It turns out it is not trivial to implement a product distribution over distributions with different event shape dimensions
    # Several classes of Pyro depend on the event shape being the same for all distributions (e.g. Independent)
    # At the moment ProductDistributions over distributions with the same event shape are enough
    # Hence the implementation is not complete

    distributions: list[Type[TorchDistribution]] = []
    def __init__(self, *distribution_parameters:list[Tensor]):

        self.distribution_instances: list[TorchDistribution] = [d(*dp) for d,dp in zip(self.distributions,distribution_parameters)]
        self.distribution_instances = [d.to_event(1) for d in self.distribution_instances]

        batch_dims = [d.batch_shape for d in self.distribution_instances]

        assert len(set(batch_dims)) == 1
        batch_dim = batch_dims[0]



    @property
    def batch_shape(self) -> torch.Size:
        return self.distribution_instances[0].batch_shape

    @property
    def event_shape(self) -> ProductEventShape:
        return ProductEventShape(*[dist.event_shape for dist in self.distribution_instances])

    def mask(self, mask) -> "ProductDistribution":
        self.distribution_instances = [dist.mask(mask) for dist in self.distribution_instances]
        return self


    def sample(self, sample_shape: torch.Size = torch.Size()) -> list[torch.Tensor]:
        sample_list = [d.sample(sample_shape) for d in self.distribution_instances]
        return sample_list

    def log_prob(self, value: list[torch.Tensor]) -> torch.Tensor:
        dist: TorchDistribution
        return sum([dist.log_prob(v) for v,dist in zip(value,self.distribution_instances)])




class ProductDistribution(TorchDistribution):
    distributions: list[Type[TorchDistribution]] = []
    def __init__(self, *distribution_parameters:list[Tensor]):

        self.distribution_instances: list[TorchDistribution] = [d(*dp) for d,dp in zip(self.distributions,distribution_parameters)]
        self.distribution_instances = [d.to_event(1) for d in self.distribution_instances]

        batch_dims = [d.batch_shape for d in self.distribution_instances]

        assert len(set(batch_dims)) == 1

        #assert lengths of event shapes are the same and coincide for all but last dimension
        event_shapes = [d.event_shape for d in self.distribution_instances]
        assert len(set([len(es) for es in event_shapes])) == 1
        assert all([es[:-1] == event_shapes[0][:-1] for es in event_shapes])

        self.event_shape = event_shapes[0][:-1]





    @property
    def batch_shape(self) -> torch.Size:
        return self.distribution_instances[0].batch_shape

    @property
    def event_shape(self) -> torch.Size:
        event_shape = self.distribution_instances[0].event_shape[:-1]
        last_dim_sum = sum(self.last_dims)
        event_shape+= torch.Size([last_dim_sum])
        return event_shape

    @property
    def last_dims(self)-> list[int]:
        return [d.event_shape[-1] for d in self.distribution_instances]
    def mask(self, mask) -> "ProductDistribution":
        self.distribution_instances = [dist.mask(mask) for dist in self.distribution_instances]
        return self


    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample_list = [d.sample(sample_shape) for d in self.distribution_instances]
        return torch.cat(sample_list,-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        #split tensor in constituents
        splits = torch.split(value,self.last_dims,-1)
        log_probs = [dist.log_prob(v) for v,dist in zip(splits,self.distribution_instances)]
        log_prob = sum(log_probs)
        return log_prob



    @event_shape.setter
    def event_shape(self, value):
        self._event_shape = value


class Multinomial1(Multinomial):

    def __init__(self,*args, total_count = 1, **kwargs):
        super().__init__(total_count, *args,**kwargs)

    def expand(self, batch_shape, _instance=None):
        super().expand(batch_shape, self)


class RandIntMultinomial(Multinomial):
    # Non hot encoded version of Multinomial
    def __init__(self, *args, total_count=1, **kwargs):
        if not isinstance(total_count, int):
            raise NotImplementedError("inhomogeneous total_count is not supported")
        self.total_count = total_count
        self._categorical = Categorical(probs=probs, logits=logits)
        self._binomial = Binomial(total_count=total_count, probs=self.probs)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        sample = super().sample(sample_shape)
        return torch.argmax(sample, dim=-1)

    def log_prob(self, value):
        return super().log_prob(torch.nn.functional.one_hot(value, num_classes=self.probs.size(-1)))


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