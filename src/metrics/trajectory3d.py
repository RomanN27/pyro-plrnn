from typing import Any

from src.metrics.metric_base import Metric, MetricLogType

from src.models.forecaster import Forecaster

from src.models.hidden_markov_model import HiddenMarkovModel

from src.models.model_sampler import ModelBasedSampler
import matplotlib.pyplot as plt

import torch
class Trajectory3D(Metric):
    log_types  = [MetricLogType.png_figure]

    def __init__(self, n_time_steps: int = 20, n_trajectories: int = 1) -> None:
        super().__init__()
        self.add_state("deterministic_trajectory", default=torch.tensor(0))
        self.add_state("stochastic_trajectories", default=torch.tensor(0))
        self.n_time_steps = n_time_steps
        self.n_trajectories = n_trajectories
    def update(self, forecaster: Forecaster, batch: torch.Tensor) -> None:

        observation_forecast_tensor, latent_forecast_tensor = forecaster(batch, self.n_time_steps, 1, probabilistic=False)
        self.deterministic_trajectory = latent_forecast_tensor

        observation_forecast_tensor, latent_forecast_tensor = forecaster(batch, self.n_time_steps, self.n_trajectories, probabilistic=True)
        self.stochastic_trajectories = latent_forecast_tensor

    def compute(self) -> Any:
        pass

    def plot(self, ax = None) -> Any:
        deterministic_trajectory = self.deterministic_trajectory.numpy()
        stochastic_trajectories = self.stochastic_trajectories.numpy()

        if ax is None:
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
        else:
            fig = ax.get_figure()
            ax1 = ax
            ax2 = fig.add_subplot(122, projection='3d')

        # Plot deterministic trajectory
        ax1.plot(deterministic_trajectory[0, :, 0], deterministic_trajectory[0, :, 1],
                 deterministic_trajectory[0, :, 2], label="Deterministic Trajectory")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()

        # Plot stochastic trajectories
        for trajectory in stochastic_trajectories:
            ax2.plot(trajectory[0, :, 0], trajectory[0, :, 1], trajectory[0, :, 2], alpha=0.5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        return fig, ax

