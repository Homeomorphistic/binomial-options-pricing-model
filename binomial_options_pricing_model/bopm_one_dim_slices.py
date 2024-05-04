"""One-dimensional slices of the price function.

They are vectorized (a loop), for easy plotting later.
Default values of constant parameters are taken from the sliders.
This way, each plot will change after change in one parameter.
"""
import bopm

import numpy as np
from typing import Callable


def vectorize(func: Callable, arg: np.ndarray) -> np.ndarray:
    """Vectorize a function which takes only one scalar argument.

    Loop over arg and obtain return for each iteraton. Return it
    as a vector.

    :param func: a function which takes only one scalar argument.
    :param arg: a vector of arguments of func.
    :returns: a vector of returns of func.
    """
    n = len(arg)
    V = np.zeros(n)
    for i in range(n):
        V[i] = func(arg[i])
    return V


def v_one_dim_slices(r, S, K, delta_t, T, sigma):
    def v_r(r: np.ndarray, american=False, call=True) -> np.ndarray:
        func = lambda arg: bopm.crr_price_option(arg, S, K, delta_t, T,sigma, american, call)[0]
        return vectorize(func, r)

    def v_s(S: np.ndarray, american=False, call=True) -> np.ndarray:
        func = lambda arg: bopm.crr_price_option(r, arg, K, delta_t, T, sigma, american, call)[0]
        return vectorize(func, S)

    def v_k(K: np.ndarray, american=False, call=True) -> np.ndarray:
        func = lambda arg: bopm.crr_price_option(r, S, arg, delta_t, T, sigma, american, call)[0]
        return vectorize(func, K)

    def v_delta_t(delta_t: np.ndarray, american=False, call=True) -> np.ndarray:
        func = lambda arg: bopm.crr_price_option(r, S, K, arg, T, sigma, american, call)[0]
        return vectorize(func, delta_t)


    def v_t(T: np.ndarray, american=False, call=True) -> np.ndarray:
        func = lambda arg: bopm.crr_price_option(r, S, K, delta_t, arg, sigma, american, call)[0]
        return vectorize(func, T)

    def v_sigma(sigma: np.ndarray, american=False, call=True) -> np.ndarray:
        func = lambda arg: bopm.crr_price_option(r, S, K, delta_t, T, arg, american, call)[0]
        return vectorize(func, sigma)

    return [v_r, v_s, v_k, v_delta_t, v_t, v_sigma]