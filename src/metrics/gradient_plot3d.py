from typing import Any

import torch

from src.metrics.metric_base import Metric, MetricLogType
from src.models.hidden_markov_model import HiddenMarkovModel
from plotly import graph_objects as go

class GradientPlot3d(Metric):
    log_types = [MetricLogType.plotly_figure]

    def __init__(self, z_dim: int = 3, z_min: float = - 3, z_max: float = 3., mesh_size: int = 20, **plotly_kwargs):
        super().__init__()
        assert z_dim == 3, "This metric only supports 3d gradients" # TODO: make this more general

        self.add_state("grads", default=torch.tensor(0.))
        self.plotly_kwargs = plotly_kwargs
        self.z_min = z_min
        self.z_max = z_max
        self.mesh_size = mesh_size
        self.z_dim = z_dim
    def update(self, hmm: HiddenMarkovModel) -> None:
        transition_model = hmm.transition_sampler.model
        mesh = self.get_mesh()

        with torch.enable_grad():
            distribution_parameters = transition_model(mesh)

        if isinstance(distribution_parameters, tuple):
            mu = distribution_parameters[0] # assuming the first element is the mean #TODO make this more robust
        else:
            mu = distribution_parameters

        mu.backward(torch.ones_like(mesh))

        grad = mesh.grad
        self.grads = grad

    def get_mesh(self):
        mesh = torch.stack(torch.meshgrid(*[torch.linspace(self.z_min, self.z_max, 20) for _ in range(self.z_dim)]),
                           -1).requires_grad_(True)
        return mesh

    def compute(self) -> None:...
    def plot(self) -> go.Figure:
        mesh = self.get_mesh()
        grad = self.grads

        x = mesh[..., 0].detach().numpy().flatten()
        y = mesh[..., 1].detach().numpy().flatten()
        z = mesh[..., 2].detach().numpy().flatten()
        u = grad[..., 0].detach().numpy().flatten()
        v = grad[..., 1].detach().numpy().flatten()
        w = grad[..., 2].detach().numpy().flatten()

        default_kwargs = {
            "sizemode": "raw",
            "sizeref": 0.3,
            "colorscale": "Portland",
            "cmin": grad.norm(dim=-1).min().item(),
            "cmax": grad.norm(dim=-1).max().item(),
            "hoverinfo": "x+y+z+u+v+w+text",
            "text": "-> gradient <-",
        }

        default_kwargs.update(self.plotly_kwargs)

        fig = go.Figure(
            data=go.Cone(
                x=x,
                y=y,
                z=z,
                u=u,
                v=v,
                w=w,
                **default_kwargs
            ),
            layout=dict(
                width=1500, height=1000, scene=dict(camera=dict(eye=dict(x=0, y=0, z=0)))
            ),
        )

        return fig
