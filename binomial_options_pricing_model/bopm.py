from typing import Callable

import numpy as np
from numpy import ndarray


def call_payoff(S: float, K: float) -> ndarray:
    """Return payoff of a call@K given asset price S_T.

    :param S: current underlying asset price.
    :param K: strike price.
    :returns: payoff of a call@K.
    """
    d = np.array(S - K)
    d[d < 0] = 0  # take only positive.
    return d


def put_payoff(S: float, K: float) -> ndarray:
    """Return payoff of a put@K given asset price S_T.

    :param S: current underlying asset price.
    :param K: strike price.
    :returns: payoff of a put@K.
    """
    d = np.array(K - S)
    d[d < 0] = 0  # take only positive.
    return d


def risk_neutral_probability(r: float, T: float,
                             u: float, d: float) -> float:
    """Return risk-neutral probability for expiry T.

    :param r: risk-free rate of the market.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :returns: risk-neutral probability at expiry T.
    """
    # TODO add d < e^rt < u constraint.
    return (np.exp(r*T) - d) / (u - d)


def expected_payoff(r: float, delta_t: float, u: float, d: float,
                    v_u: ndarray[float], v_d: ndarray[float]) -> ndarray[float]:
    """Return risk-neutral probability for expiry T.

        :param r: risk-free rate of the market.
        :param delta_t: time step.
        :param u: price up-scaling factor.
        :param d: price down-scaling factor.
        :param v_d: value of an option at down child.
        :param v_u: value of an option at upper child.
        :returns: value of an option at present node.
    """
    p = risk_neutral_probability(r, delta_t, u, d)
    return np.exp(-r*delta_t) * (p * v_u + (1-p) * v_d)


def binomial_tree(S: float, K: float, delta_t: float, T: float,
                  u: float, d: float, only_leafs: bool = True) -> ndarray:
    """Produce a vector of the last level of the binomial tree.

    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :param only_leafs: compute only leafs or whole tree?
    :returns: vector of the last level of the binomial tree or the whole tree.
    """
    n = int(T / delta_t)
    i = np.arange(n + 1)  # i = 0, 1, ..., n -1, n
    j = n - i  # j = n, n-1, ..., 1, 0
    S = S * np.ones(n + 1)
    leafs = S * (u**i) * (d**j)  # leafs = S * u^i * d^(n-i)

    if only_leafs:
        return leafs
    else:
        tree = [leafs]
        for k in range(n):
            # for each level of the tree we have less nodes (and steps) to reach them
            i = np.delete(i, n - k)
            j = np.delete(j, 0)
            S = np.delete(S, 0)
            tree.append(S * (u ** i) * (d ** j))

        tree.reverse()
        return tree


def price_european_option(r: float, S: float, K: float,
                          delta_t: float, T: float,
                          u: float, d: float,
                          payoff: Callable[..., ndarray[float]] = call_payoff) \
        -> tuple[ndarray[float], list]:
    """Price a european option.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :param payoff: payoff function of an option.
    :returns: price of an option.
    """
    # Get the last level of binomial tree.
    n = int(T / delta_t)
    # If the option is european, we need only leafs.
    v = binomial_tree(S, K, delta_t, T, u, d, only_leafs=True)
    v = payoff(v, K)
    # Keep each level of the tree.
    history = [v]

    for i in range(n):
        v_u = np.delete(v, 0)  # remove first, so that we have v_i+1
        v_d = np.delete(v, n-i)  # remove last, so that we have v_i,
        v = expected_payoff(r, delta_t, u, d, v_u, v_d)
        history.append(v)

    return v[0], history  # [0] to unpack array with one element


def price_european_put(r: float, S: float, K: float,
                          delta_t: float, T: float,
                          u: float, d: float) \
        -> tuple[ndarray[float], list]:
    """Price an european put option.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :returns: price of an option.
    """
    return price_european_option(r, S, K, delta_t, T, u, d, payoff=put_payoff)


def price_european_call(r: float, S: float, K: float,
                          delta_t: float, T: float,
                          u: float, d: float) \
        -> tuple[ndarray[float], list]:
    """Price an european call option.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :returns: price of an option.
    """
    return price_european_option(r, S, K, delta_t, T, u, d, payoff=call_payoff)


def price_american_option(r: float, S: float, K: float,
                          delta_t: float, T: float,
                          u: float, d: float,
                          payoff: Callable[..., ndarray[float]] = put_payoff) \
        -> tuple[ndarray[float], list]:
    """Price an american option.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :param payoff: payoff function of an option.
    :returns: price of an option.
    """
    # Get the last level of binomial tree.
    n = int(T / delta_t)
    # If the option is american, get the full tree.
    tree = binomial_tree(S, K, delta_t, T, u, d, only_leafs=False)
    value = payoff(tree[n], K)
    # Keep each level of the tree.
    history = [value]

    for i in range(n):
        value_up = np.delete(value, 0)  # remove first, so that we have v_i+1
        value_down = np.delete(value, n-i)  # remove last, so that we have v_i,
        payoffs = payoff(tree[n - i - 1], K)
        expected_payoffs = expected_payoff(r, delta_t, u, d, value_up, value_down)

        # Get the expected and actual payoffs
        profitability = np.vstack((expected_payoffs, payoffs))
        # Is exercising an option is more profitable than expected payoff?
        value = np.max(profitability, axis=0)
        history.append(value)

    return value[0], history  # [0] to unpack array with one element


def price_american_put(r: float, S: float, K: float,
                          delta_t: float, T: float,
                          u: float, d: float) \
        -> tuple[ndarray[float], list]:
    """Price an american put option.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :returns: price of an option.
    """
    return price_american_option(r, S, K, delta_t, T, u, d, payoff=put_payoff)


def price_american_call(r: float, S: float, K: float,
                          delta_t: float, T: float,
                          u: float, d: float) \
        -> tuple[ndarray[float], list]:
    """Price an american call option.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :returns: price of an option.
    """
    return price_american_option(r, S, K, delta_t, T, u, d, payoff=call_payoff)


if __name__ == "__main__":
    delta_t = 3/12#1/12
    T = 1/2#2
    n = int(T / delta_t)
    K = 21#48
    sigma = .3
    u = 1.1#np.exp(sigma * np.sqrt(delta_t))
    d = 0.9#1/u
    r = .12#.02
    S = 20#50
    v, h = price_american_option(r, S, K, delta_t, T, u, d)
    print(h)