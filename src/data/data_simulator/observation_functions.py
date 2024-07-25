import numpy as np
from scipy.signal import convolve

x_component_observation = lambda trajectory: trajectory[...,:1]

class Downsampler:

    def __init__(self,n_of_target_points:int):
        self.n_of_target_points=n_of_target_points

    def __call__(self, x: np.ndarray):
        m = x.shape[0] // self.n_of_target_points
        return x[::m][:self.n_of_target_points]

standard_downsampler = Downsampler(512)

default_obs = lambda x: standard_downsampler(x_component_observation(x))

class HemodynamicResponseFunction:
    def __init__(self, tr=2.0, duration=32.0):
        self.tr = tr
        self.duration = duration
        self.hrf: np.ndarray = self._create_hrf()

    def _gamma_pdf(self, x, shape, scale):
        return (x ** (shape - 1) * np.exp(-x / scale)) / (scale ** shape * np.math.gamma(shape))

    def _create_hrf(self) -> np.array:
        dt = self.tr / 16.0
        time_points = np.arange(0, self.duration, dt)
        hrf = self._gamma_pdf(time_points, 6, 1) - 0.35 * self._gamma_pdf(time_points, 12, 1)
        hrf /= np.sum(hrf)  # Normalize the HRF
        return hrf

    def __call__(self, signal: np.ndarray):
        #assuming first corresponds to timedimension
        if len(signal) < len(self.hrf):
            raise ValueError("Input signal is shorter than the HRF duration.")
        repeated_hrf = self.hrf.reshape(-1,1).repeat(signal.shape[1],1)
        convolved_signal = convolve(signal, repeated_hrf, mode='full')[:len(signal)]
        return convolved_signal

standard_hrf = HemodynamicResponseFunction()
if __name__ == "__main__":
    # Example usage:
    # Create an HRF object
    hrf = HemodynamicResponseFunction(tr=2.0, duration=32.0)

    # Example input signal (e.g., neural activity over time)
    input_signal = np.random.rand(1000,17)

    # Perform convolution
    convolved_signal = hrf(input_signal)

    import matplotlib.pyplot as plt

    plt.plot(input_signal)
    plt.plot(convolved_signal)
    plt.show()