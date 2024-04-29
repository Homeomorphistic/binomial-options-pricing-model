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
                    v_u: float, v_d: float) -> float:
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


def binomial_leafs(S: float, K: float, delta_t: float, T: float,
                   u: float, d: float) -> ndarray:
    """Produce a vector of the last level of the binomial tree.

    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :returns: vector of the last level of the binomial tree.
    """
    n = int(T / delta_t)
    i = np.arange(n + 1)  # i = 0, 1, ..., n -1, n
    j = n - i  # j = n, n-1, ..., 1, 0
    V = S * np.ones(n + 1)
    V = V * (u**i) * (d**j)  # V = S * u^i * d^(n-i)
    return V


def price_option(r: float, S: float, K: float,
                 delta_t: float, T: float,
                 u: float, d: float) -> float:
    """Price an option.

    :param r: risk-free rate of the market.
    :param S: current underlying asset price.
    :param K: strike price.
    :param delta_t: time step.
    :param T: expiration date of an option in years.
    :param u: price up-scaling factor.
    :param d: price down-scaling factor.
    :returns: price of an option.
    """
    # Get the last level of binomial tree.
    n = int(T / delta_t)
    v = binomial_leafs(S, K, delta_t, T, u, d)
    v = call_payoff(v, K)

    for i in range(n):
        v_u = np.delete(v, 0)
        v_d = np.delete(v, n-i)
        print(v)
        v = expected_payoff(r, delta_t, u, d, v_u, v_d)

    print(v)
    return v

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
    price_option(r, S, K, delta_t, T, u, d)
