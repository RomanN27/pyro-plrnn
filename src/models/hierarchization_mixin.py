import torch.nn as nn
import numpy as np
import pyro
from src.models.transition_models.plrnns.plrnn_base import PLRNN
s = nn.Sequential(nn.Linear(10,12), nn.Linear(12,10))
s.state_dict()
from copy import deepcopy
import torch
import copy
class StochasticHierarchizationMixin:

    def __init__(self: nn.Module, n_latent_factors:int ,*args,**kwargs):
        super().__init__(*args,**kwargs)
        parameter_dict = self.state_dict()
        self.parameter_shapes = {k:v.shape for k,v in parameter_dict.items()}
        self.n_latent_factors = n_latent_factors
        self.projection_models = {parameter_name: nn.Linear(n_latent_factors, np.prod(shape)) for parameter_name, shape in self.parameter_shapes.items()}
    def forward(self:nn.Module, z):
        latent_factors = pyro.sample("latent_factors", pyro.distributions.Normal(0,1).expand([z.size(0),self.n_latent_factors]))
        state_dicts = [{} for _ in range(z.size(0))]
        for parameter_name, projection_model in self.projection_models.items():
            parameter_shape = self.parameter_shapes[parameter_name]
            parameters = [projection_model(latent_factors_i) for latent_factors_i in latent_factors.split(1)]
            for state_dict, parameter in zip(state_dicts, parameters):
                state_dict[parameter_name] = parameter.view(parameter_shape)

        results = []

        for state_dict in state_dicts:
            self.load_state_dict(state_dict)
            results.append(super().forward(z))

        concatenated_results = torch.stack(results)
        return concatenated_results


import torch
import torch.nn as nn


class DynamicEmbedding(nn.Module):
    def __init__(self, embedding_dim, initializer=torch.nn.init.normal_):
        """
        Args:
            embedding_dim (int): Dimension of the embedding vector for each index.
            initializer (callable): Function to initialize the embeddings. Defaults to normal distribution.
        """
        super(DynamicEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.initializer = initializer
        self.embeddings = {}

    def forward(self, indices : list):
        """
        Args:
            indices (torch.Tensor): A tensor of indices for which to fetch or initialize embeddings.

        Returns:
            torch.Tensor: A tensor of shape (num_indices, embedding_dim) containing the embeddings.
        """
        output = []
        for index in indices: # Convert to Python int for dictionary keys
            if index not in self.embeddings:
                # Initialize the embedding on the fly if not available
                self.embeddings[index] = torch.empty(self.embedding_dim,requires_grad=True)
                self.initializer(self.embeddings[index])

            # Append the embedding corresponding to the index
            output.append(self.embeddings[index])

        # Stack the output embeddings into a tensor
        return torch.stack(output, dim=0)


class DeterministicHierarchizationMixin:

class DeterministicHierarchizationMixin:

    def __init__(self: nn.Module, n_latent_factors:int ,initializer=torch.nn.init.normal_, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tmp_model = copy.deepcopy(self)
        parameter_dict = self.state_dict()
        self.parameter_shapes = {k:v.shape for k,v in parameter_dict.items()}
        self.n_latent_factors = n_latent_factors
        self.embedding = DynamicEmbedding(n_latent_factors, initializer)
        self.projection_models = {parameter_name: nn.Linear(n_latent_factors, np.prod(shape)) for parameter_name, shape in self.parameter_shapes.items()}
    def forward(self:nn.Module, z,indices: list[int]):
        latent_factors = self.embedding(indices)
        state_dicts = [{} for _ in range(z.size(0))]

        for parameter_name, projection_model in self.projection_models.items():
            parameter_shape = self.parameter_shapes[parameter_name]
            parameters = [projection_model(latent_factors_i) for latent_factors_i in latent_factors.split(1)]
            for state_dict, parameter in zip(state_dicts, parameters):
                state_dict[parameter_name] = parameter.view(parameter_shape)

        results = []

        for state_dict in state_dicts:
            tmp_model = copy.deepcopy(self)
            tmp_model.load_state_dict(state_dict)
            results.append(super(DeterministicHierarchizationMixin,tmp_model).forward(z))

        concatenated_results = torch.stack(results)
        return concatenated_results

if __name__ == '__main__':

    class VanillaSequential(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(10,12)
            self.lin2 = nn.Linear(12,10)
        def forward(self, x):
            x = self.lin1(x)
            x = self.lin2(x)
            return x

    class DeterministicHierarchizedSequential(DeterministicHierarchizationMixin, VanillaSequential):
        pass

    #model  = DeterministicHierarchizedSequential(5)
    z = torch.randn(3,10)
    #indices = [1,2,3]
    #result = model(z,indices)
    #loss =  result.sum()
    #loss.backward()

    #print(model.embedding.embeddings)

    linear = nn.Linear(10,12)
    latent_factors  = torch.randn(3,5,requires_grad=True)



    weight_projector  = nn.Linear(5,120)
    bias_projector = nn.Linear(5,12)
    from torch.autograd.functional import jacobian

    jac_weight = jacobian(weight_projector,latent_factors)
    jac_bias  = jacobian(bias_projector,latent_factors)
    weight_grads = []
    bias_grads = []
    for latent_factor in latent_factors.split(1):
        weight = weight_projector(latent_factor).view(12,10)
        bias = bias_projector(latent_factor).view(12)

        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)


        output = linear(z)
        print(output)
        loss = output.sum()
        print(loss)
        loss.backward()


        weight_grads.append(linear.weight.grad.clone().detach())
        bias_grads.append(copy.deepcopy(linear.bias.grad.clone().detach()))


        linear.weight.grad.zero_()
        linear.bias.grad.zero_()

    latent_factors_grad = jac_weight @ linear.weight.grad  + linear.bias.grad @ jac_bias



    input_ = torch.randn(3,10)
    #output_1
    linear.load_state_dict(new_state_dict_1)
    output_1 = linear(input_)

    #output_2
    linear.load_state_dict(new_state_dict_2)
    output_2 = linear(input_)

    loss = output_1.sum() + output_2.sum()
    loss.backward()

