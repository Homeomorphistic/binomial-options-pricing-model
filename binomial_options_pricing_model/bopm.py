"""Binomial options pricing model.

This script contains functions that build up a binomial options pricing model,
formalized by Cox, Ross and Rubinstein (1979).

"""
from typing import Callable, Any

import numpy as np
from numpy import ndarray


def call_payoff(S: ndarray, K: float) -> ndarray:
    """Return payoff of a call@K given asset price S_T.

    :param S: current underlying asset price.
    :param K: strike price.
    :returns: payoff of a call@K.
    """
    payoff = np.array(S - K)
    payoff[payoff < 0] = 0  # take only positive.
    return payoff


def put_payoff(S: ndarray, K: float) -> ndarray:
    """Return payoff of a put@K given asset price S_T.

    :param S: current underlying asset price.
    :param K: strike price.
    :returns: payoff of a put@K.
    """
    payoff = np.array(K - S)
    payoff[payoff < 0] = 0  # take only positive.
    return payoff


def risk_neutral_probability(r: float, t: float,
                             up: float, down: float) -> float:
    """Return risk-neutral probability for a period of t.

    :param r: risk-free rate of the market.
    :param t: time period (in years).
    :param up: price up-scaling factor.
    :param down: price down-scaling factor.
    :returns: risk-neutral probability for a period of t.
    """
    # d < e^rt < u constraint makes sure it's a probability.
    if down < np.exp(r * t) < up:
        return (np.exp(r * t) - down) / (up - down)
    else:
        raise ValueError("up or down scaling factor is out of constraints.")


def expected_payoff(r: float, t: float, up: float, down: float, p: float,
                    value_up: ndarray, value_down: ndarray) -> ndarray:
    """Return expected payoff of an option.

    :param r: risk-free rate of the market.
    :param t: time period (in years).
    :param up: price up-scaling factor.
    :param down: price down-scaling factor.
    :param p: risk-neutral probability.
    :param value_down: value of an option at down child.
    :param value_up: value of an option at upper child.
    :returns: expected payoff of an option at present node.
    """
    return np.exp(-r * t) * (p * value_up + (1 - p) * value_down)


def price_binomial_tree(S: float, delta_t: float, T: float,
                        up: float, down: float) -> list[ndarray]:
    """Produce a price binomial tree.

    It produces binomial tree, with asset prices that can go only up (S*up)
    or down (S*down) after delta_t time.
    The tree is represented by the list containing each level of the tree.

    :param S: current underlying asset price.
    :param delta_t: time step (in years).
    :param T: maturity of an option (in years).
    :param up: price up-scaling factor.
    :param down: price down-scaling factor.
    :returns: list of binomial tree levels.
    """
    # Number of levels of a binomial tree.
    n = int(T / delta_t)

    i = np.arange(n + 1)  # i = 0, 1, ..., n - 1, n
    j = n - i  # j = n, n-1, ..., 1, 0
    S = S * np.ones(n + 1)
    leafs = S * (up ** i) * (down ** j)  # leafs = S * up^i * down^(n-i)

    tree = [leafs]
    for k in range(n):
        # for each level of the tree we have less nodes (and steps) to reach them
        i = np.delete(i, n - k)  # i = 0, 1, ..., n-k-1, n-k
        j = np.delete(j, 0)  # j = n-k, n-k-1, ..., 1, 0
        S = np.delete(S, 0)  # remove to match the length of i and j
        tree.append(S * (up ** i) * (down ** j))

    tree.reverse()
    return tree


def price_option(r: float, S: float, K: float, delta_t: float,
                 T: float, up: float, down: float,
                 american: bool = False, call: bool = True) -> tuple[ndarray, list[ndarray]]:
    """Price an option@K with maturity T.

    Prices a european/american vanilla options given the underlying
    asset and market parameters. We assume discrete time steps and a
    lattice of available prices for the asset.
    It returns a tuple containing present price and a binomial tree
    with option prices at each node.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step (in years).
    :param T: maturity of an option (in years).
    :param up: price up-scaling factor.
    :param down: price down-scaling factor.
    :param american: is it american option?
    :param call: is it a call option? Otherwise, put.
    :returns: tuple of price of an option and a binomial tree.
    """
    payoff = call_payoff if call else put_payoff
    # Number of levels of a binomial tree.
    n = int(T / delta_t)
    # Risk-neutral probability.
    p = risk_neutral_probability(r, delta_t, up, down)
    # Price binomial tree.
    tree = price_binomial_tree(S, delta_t, T, up, down)
    # Values of an option at leafs (last level of the tree).
    value = payoff(tree[n], K)
    # Store each level of the tree.
    history = [value]

    for i in range(n):
        value_up = np.delete(value, 0)  # remove first, so that we have v_i+1
        value_down = np.delete(value, n-i)  # remove last, so that we have v_i,
        value = expected_payoff(r, delta_t, up, down, p, value_up, value_down)

        if american:  # if american, check if it's profitable to exercise option.
            # Expected and actual payoffs at current tree level.
            payoffs = payoff(tree[n - i - 1], K)
            expected_payoffs = expected_payoff(r, delta_t, up, down, p, value_up, value_down)
            profitability = np.vstack((expected_payoffs, payoffs))
            # Is exercising an option more profitable than expected payoff?
            value = np.max(profitability, axis=0)

        history.append(value)

    history.reverse()
    return value[0], history  # [0] to unpack array with one element


def crr_price_option(r: float, S: float, K: float, delta_t: float, T: float,
                     sigma: float, american: bool = False, call: bool = True) -> tuple[ndarray, list]:
    """Price a option@K with maturity T.

    Prices a european/american call/put option given the underlying asset and market parameters.
    We assume discrete time steps and a lattice of available prices for the asset.

    Use Cox, Ross and Rubinstein method to obtain up and down factors
    through volatility of an asset (sigma):
    up = e^(sigma * sqrt(delta_y)) = 1/d

    It returns a tuple containing present price and a binomial tree
    with option prices at each node.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step (in years).
    :param T: maturity of an option (in years).
    :param sigma: volatility of an asset.
    :param american: is it american option?
    :param call: is it a call option? Otherwise, put.
    :returns: tuple of price of an option and a binomial tree.
    """
    up = np.exp(sigma * np.sqrt(delta_t))
    down = 1/up
    return price_option(r, S, K, delta_t, T, up, down, american, call)


if __name__ == "__main__":
    r, S, K, delta_t, T, sigma = .02, 50, 40, 1/12, 2, .3

