"""One-dimensional slices of the price function.

Price of an option (in CRR model) is a multivariable function: V(r, S_0, K, delta_t, T, sigma).
First step in analyzing price of an option is to look at one-dimensional slices of price function.

This script contains functions used later for plotting:
V(r)$, V(S_0), V(K), V(delta_t), V(T), V(sigma).
They are vectorized (a loop), for easy plotting later.

IMPORTANT: the purpose of this script is only for testing static plots
(without interactive sliders)
"""
import binomial_options_pricing_model.bopm as bopm

import numpy as np
from typing import Callable, Tuple


def vectorize(func: Callable, arg: np.ndarray) -> np.ndarray:
    """Vectorize a function which takes only one scalar argument.

    Loop over arg and obtain return for each iteration. Return it
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


def crr_price_functions(r: float, S: float, K: float,
                        delta_t: float, T: float, sigma: float) -> Tuple[Callable]:
    """Return price functions in CRR model.

    :param r: default risk-free rate of the market.
    :param S: default current underlying asset price.
    :param K: default strike price.
    :param delta_t: default time step (in years).
    :param T: default maturity of an option (in years).
    :param sigma: default volatility of an asset.
    :returns: tuple of price functions of each parameter.
    """
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

    return v_r, v_s, v_k, v_delta_t, v_t, v_sigma