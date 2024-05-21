from typing import Tuple

import numpy as np
from numpy import ndarray as array_type

""" 
If fractional Brownian motion samples get passed along, they should have the following type. The 1st component is
reserved for the Hurst parameter, the 2nd for the time space and the 3rd for the sample matrix. 
"""
fBmType: type = Tuple[float, array_type, array_type]


def fractional_brownian_motion(H: float, n: int, m: int, t_max: float = 1) -> fBmType:
    """
    Returns a sample path simulations of m independent fractional Brownian motions on the interval [0, tMax] with a grid
    size of N. Also returns the array of ticks.
    :param H: Hurst parameter.
    :param n: Number of Grid points.
    :param m: Number of independent fBms.
    :param t_max: Time horizon.
    :return: A tuple (H, T, M) where H is the Hurst parameter, T are the ticks and M are the sampled paths from 0 to
    t_max in form of a (m, n) matrix.
    """
    assert 0 <= H <= 1

    if H == 0.0:
        B = np.random.randn(m, n)
        B -= B[:, 0].reshape(m, 1)
        B /= np.sqrt(2)
        return H, np.linspace(0, t_max, n), B
    elif H == 1.0:
        Xi = np.random.randn(m, 1)
        T = np.linspace(0, t_max, n)
        return H, T, T.reshape(1, n) * Xi
    else:
        def from_discrete_fractional_noise__(dfn_: array_type, H_: float, N_: int, m_: int, tMax_: float) -> fBmType:
            T_ = np.linspace(0, tMax_, N_)
            samples_ = np.hstack((np.zeros((m_, 1)), dfn_.cumsum(axis=1) * ((N_ - 1) / tMax_) ** (-H_)))
            return H_, T_, samples_

        n_ = n - 1
        j = np.arange(2 * n_)
        gamma = 0.0 * np.ones(n_ + 1)
        gamma[1:n_] = 0.5 * (j[0:n_ - 1] ** (2 * H) - 2 * j[1:n_] ** (2 * H) + j[2:n_ + 1] ** (2 * H))
        gamma[0] = 1.0
        lam = np.fft.irfft(gamma)[:n_ + 1] * complex(1,0)
        W = np.empty((m, n_ + 1), dtype='complex_')
        W[...] = np.random.randn(m, n_ + 1)
        W[:, 1:n_] = (W[:, 1:n_] + complex(0, 1) * np.random.randn(m, n_ - 1)) / np.sqrt(2)
        dfn = np.fft.hfft(W * np.sqrt(lam))[:, :n_]

        return from_discrete_fractional_noise__(dfn, H, n, m, t_max)


def time_projection(time, path):
    return time


def path_projection(time, path):
    return path


def augmented_paths(T=1.0, steps=100, MC=10000, H=0.5, path_functions=None):
    if path_functions is None:
        path_functions = [time_projection, path_projection]

    out_dimension = len(path_functions)

    if (H >= 0) and (H < 1):
        _, time, B = fractional_brownian_motion(H, steps + 1, MC, T)
    elif H == 1:
        N = np.random.randn(MC).reshape((MC, 1))
        time = np.linspace(0, T, steps + 1)
        B = N * time.reshape((1, steps + 1))
    else:
        raise AttributeError("H should be in [0,1].")

    X_ = np.empty((MC, steps + 1, out_dimension))
    for i, f in enumerate(path_functions):
        X_[:, :, i] = f(time, B)

    return time, B, X_