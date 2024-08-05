from src.data.data_simulator.data_simulator import DataSimulator

from src.data.data_simulator.drift_functions import lorentz_drift, roessler_drift
from src.data.data_simulator.observation_functions import Slicer, Downsampler
class LorenzSimulator(DataSimulator):
    def __init__(self, initial_state: list[float], t_range: tuple[float, float] = (0, 50), dt: float = 0.01, n_observed_dim:int = 1,
                 sampling_length:int = 512):
        slicer_observation_function = Slicer(n_observed_dim)
        standard_downsampler = Downsampler(sampling_length)
        observation_function = lambda x: standard_downsampler(slicer_observation_function(x))
        super().__init__(lorentz_drift, initial_state, t_range, dt, observation_function=observation_function)

    @staticmethod
    def hydra_get_data(initial_state: list[float], n_observed_dim: int, sampling_length: int, t_range: tuple[float, float] = (0, 50), dt: float = 0.01,*args, **kwargs):
        sim = LorenzSimulator(initial_state, t_range, dt, n_observed_dim, sampling_length)
        return sim.get_data()