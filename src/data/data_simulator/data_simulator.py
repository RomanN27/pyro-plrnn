import numpy as np
import sdeint
from typing import Callable, ParamSpec, Generic, Optional, TypeVar, Iterable

from contextlib import contextmanager
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import torch

from src.data.data_simulator.observation_functions import default_obs, standard_hrf, standard_downsampler

simple_noise = lambda state, t: np.eye(3) * 1
simple_gaussian_noise = lambda trajectory: np.random.normal(trajectory, scale=0.1)

SP = ParamSpec("SP")


class DataSimulator:

    def __init__(self, drift_function: Callable[[np.array, float, SP], np.array],
                 initial_state: list[float], drift_parameters: dict = None, t_range: tuple[float, float] = (0, 50),
                 dt: float = 0.01, noise_function: Callable[[np.array, float], np.array] = simple_noise,
                 observation_function: Callable[[np.array], np.array] = default_obs
                 , observation_noise: Callable[[np.array], np.array] = simple_gaussian_noise):
        if drift_parameters is None:
            drift_parameters = {}
        self.drift_parameters = drift_parameters
        self.drift_function = drift_function
        self.initial_state = initial_state
        self.noise_function = noise_function
        self.t_range = t_range
        self.dt = dt
        self.observation_function = observation_function
        self.observation_noise = observation_noise

    def solve(self):
        t = np.arange(self.t_range[0], self.t_range[1], self.dt)
        result = sdeint.itoint(self.drift_function, self.noise_function, self.initial_state, t)
        return t, result

    def run_system(self) -> np.ndarray:
        _, result = self.solve()
        observed_result = self.observation_function(result)
        noised_observations = self.observation_noise(observed_result)
        return noised_observations

    def set(self, name: str, value) -> "DataSimulator":
        if name not in self.__dict__:
            if name not in self.drift_parameters:
                raise Exception(f"{name} is not valid")
            else:
                self.drift_parameters.update({name: value})
        else:
            setattr(self, name, value)

        return self

    def set_multiple(self, name_value_dict: dict):
        for name, value in name_value_dict:
            self.set(name, value)
        return self

    def get_data(self, combinations: Iterable[dict],
                 collate_fn: Callable[[list[np.array]], torch.Tensor] = torch.stack) -> torch.Tensor:
        data = []
        for combination in combinations:
            observed_data = self.set_multiple(combination).run_system()
            data.append(observed_data)
        tensor = collate_fn(data)
        return tensor

    #convenience function to call this from hydra


if __name__ == "__main__":
    from src.data.data_simulator.drift_functions import lorentz_drift, roessler_drift

    initial_state = [1, 1, 1]
    lorentz_simulator = DataSimulator(lorentz_drift, initial_state, observation_function=lambda x: x,
                                      noise_function=lambda x, t: np.ones((3, 1)) * 0.05 * x)
    sol = lorentz_simulator.run_system()

    # Plotting the solution
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Stochastic Lorenz System Solution with Brownian Noise")
    plt.show()
