import numpy as np


def call_payoff(S: float, K: float) -> float:
    """Return payoff of a call@K given asset price S_T.

    :param S: current underlying asset price.
    :param K: strike price.
    :returns: payoff of a call@K.
    """
    return max(S - K, 0)


def put_payoff(S: float, K: float) -> float:
    """Return payoff of a put@K given asset price S_T.

    :param S: current underlying asset price.
    :param K: strike price.
    :returns: payoff of a put@K.
    """
    return max(K - S, 0)


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
        :param delta_t: expiration date of an option in years.
        :param u: price up-scaling factor.
        :param d: price down-scaling factor.
        :param v_d: value of an option at down child.
        :param v_u: value of an option at upper child.
        :returns: value of an option at present node.
    """
    p = risk_neutral_probability(r, delta_t, u, d)
    return np.exp(-r*delta_t) * (p * v_u + (1-p) * v_d)
