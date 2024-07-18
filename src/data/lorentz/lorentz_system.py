import numpy as np
import sdeint
from typing import Callable
from contextlib import contextmanager
import matplotlib.pyplot as plt

from src.data.lorentz.observation_functions import default_obs, standard_hrf, standard_downsampler

simple_noise = lambda state,t : np.eye(3)*1
simple_gaussian_noise = lambda trajectory :np.random.normal(trajectory,scale=0.1)
class GeneralLorentzSystem:

    def __init__(self,sigma: float =10.0,rho: float=28.0,beta: float=8.0/3.0,initial_state:list[float] = [1.0, 1.0, 1.0],t_range:tuple[float,float] =(0,50),
                 dt:float =0.01,noise_function:Callable[[np.array,float],np.array]=simple_noise,observation_function:Callable[[np.array],np.array] = default_obs
                 ,observation_noise: Callable[[np.array],np.array]=simple_gaussian_noise):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initial_state = initial_state
        self.noise_function = noise_function
        self.t_range = t_range
        self.dt = dt
        self.observation_function = observation_function
        self.observation_noise = observation_noise

    def set(self,name:str,value):
        self.__setattr__(name,value)
        return self


    def lorenz_sde(self,state, t):
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])


    def solve_lorenz_sde(self):
        t = np.arange(self.t_range[0], self.t_range[1], self.dt)
        # Use partial to bind parameters
        lorenz_sde_partial = self.lorenz_sde
        result = sdeint.itoint(lorenz_sde_partial, self.noise_function, self.initial_state, t)
        return t, result

    def run_system(self)->np.ndarray:
        _, result = self.solve_lorenz_sde()
        observed_result = self.observation_function(result)
        noised_observations = self.observation_noise(observed_result)
        return noised_observations

if __name__ == "__main__":
    sol = GeneralLorentzSystem(observation_function=standard_downsampler).run_system()


    # Plotting the solution
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Stochastic Lorenz System Solution with Brownian Noise")
    plt.show()

