import numpy as np


def lorentz_drift(state, t, sigma: float = 10.0, rho: float = 28.0, beta:float = 8.0/3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def roessler_drift(state, t, a:float  =0.2, b: float = 0.2, c: float = 5.7):
    x, y, z = state
    dxdt = - (y + z)
    dydt = x + a * y
    dzdt = b + (x - c) * z
    return np.array([dxdt, dydt, dzdt])
