from typing import Any

from src.metrics.metric_base import Metric, MetricLogType

from src.models.forecaster import Forecaster
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.hidden_markov_model import HiddenMarkovModel

from src.models.model_sampler import ModelBasedSampler
import matplotlib.pyplot as plt

import torch
class Trajectory3D(Metric):
    log_types  = [MetricLogType.plotly_figure]

    def __init__(self, n_time_steps: int = 20, n_trajectories: int = 1) -> None:
        super().__init__()
        self.add_state("deterministic_trajectory", default=torch.tensor(0))
        self.add_state("stochastic_trajectories", default=torch.tensor(0))
        self.add_state("ground_truth", default=torch.tensor(0))
        self.n_time_steps = n_time_steps
        self.n_trajectories = n_trajectories
    def update(self, forecaster: Forecaster, batch: torch.Tensor) -> None:

        observation_forecast_tensor, latent_forecast_tensor = forecaster(batch, self.n_time_steps, 1, probabilistic=False)
        self.deterministic_trajectory = latent_forecast_tensor

        observation_forecast_tensor, latent_forecast_tensor = forecaster(batch, self.n_time_steps, self.n_trajectories, probabilistic=True)
        self.stochastic_trajectories = latent_forecast_tensor

        self.ground_truth = batch


    def compute(self) -> Any:
        pass

    def plot(self, ax = None) -> Any:
        deterministic_trajectory = self.deterministic_trajectory.numpy()
        stochastic_trajectories = self.stochastic_trajectories.numpy()
        ground_truth = self.ground_truth.numpy()

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot deterministic trajectory
        fig.add_trace(
            go.Scatter3d(
                x=deterministic_trajectory[0, :, 0],
                y=deterministic_trajectory[0, :, 1],
                z=deterministic_trajectory[0, :, 2],
                mode='lines',
                name='Deterministic Trajectory'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=ground_truth[0, :, 0],
                y=ground_truth[0, :, 1],
                z=ground_truth[0, :, 2],
                mode='lines',
                name='Ground Truth',
                opacity=0.5
            ),
            row=1, col=1
        )

        # Plot stochastic trajectories
        for trajectory in stochastic_trajectories:
            fig.add_trace(
                go.Scatter3d(
                    x=trajectory[0, :, 0],
                    y=trajectory[0, :, 1],
                    z=trajectory[0, :, 2],
                    mode='lines',
                    opacity=0.5
                ),
                row=1, col=2
            )

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene2=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        return fig

