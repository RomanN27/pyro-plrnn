import torch
from torch import nn as nn

from src.models.model_sampler import ModelBasedSampler


class SkewProduct(nn.Module):

    def __init__(self, *autonomous_transition_models: ModelBasedSampler, target_model: ModelBasedSampler):
        super().__init__()
        self.autonomous_transition_models = autonomous_transition_models
        self.target_model = target_model
        self.z_dims = [transition_model.model.z_dim for transition_model in autonomous_transition_models]
        self.free_target_dims = target_model.model.z_dim - sum(self.z_dims)
        assert self.free_target_dims > 0
        self.z_dims.append(self.free_target_dims)
        self.z_dim = sum(self.z_dims)

    def forward(self, z):
        z_chunks = list(torch.split(z,self.z_dims,-1))
        target_chunk = z_chunks.pop()
        sub_space_new_z = torch.cat([m(z_chunk) for m,z_chunk in zip(self.autonomous_transition_models,z_chunks)],dim=-1)
        updated_z = torch.cat([sub_space_new_z,target_chunk],-1)
        new_z = self.target_model(updated_z)
        final_z = torch.cat([sub_space_new_z,new_z[...,-self.free_target_dims:]],-1)
        return final_z
