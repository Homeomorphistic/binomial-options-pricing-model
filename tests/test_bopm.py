import numpy as np
from numpy.testing import assert_allclose
import pytest

import binomial_options_pricing_model.bopm as bopm


@pytest.mark.parametrize("S, K, payoff",
                         [(22, 21, 1), (18, 21, 0)])
def test_call_payoff(S, K, payoff):
    assert bopm.call_payoff(S, K) == payoff


@pytest.mark.parametrize("S, K, payoff",
                         [(22, 21, 0), (18, 21, 3)])
def test_put_payoff(S, K, payoff):
    assert bopm.put_payoff(S, K) == payoff


@pytest.mark.parametrize("r, T, u, d, prob",
                         [(.12, 3/12, 1.1, 0.9, .6523)])
def test_risk_neutral_probability(r, T, u, d, prob):
    return assert_allclose(actual=bopm.risk_neutral_probability(r, T, u, d),
                           desired=prob, rtol=1e-4)


@pytest.mark.parametrize("r, delta_t, u, d, v_u, v_d, v",
                         [(.12, 3/12, 1.1, 0.9, 3.2, 0, 2.0257),
                          (.12, 3/12, 1.1, 0.9, 0, 0, 0),
                          (.12, 3/12, 1.1, 0.9, 2.0257, 0, 1.28)])
def test_expected_payoff(r, delta_t, u, d, v_u, v_d, v):
    return assert_allclose(actual=bopm.expected_payoff(r, delta_t, u, d, v_u, v_d),
                           desired=v, rtol=1e-2)


@pytest.mark.parametrize("S, K, delta_t, T, u, d, leafs",
                         [(20, 21, 3/12, 1/2, 1.1, 0.9, np.array([16.2, 19.8, 24.2])),
                          (50, 52, 1, 2, 1.2, 0.8, np.array([32, 48, 72]))
                          ])
def test_binomial_leafs(S, K, delta_t, T, u, d, leafs):
    return assert_allclose(actual=bopm.binomial_leafs(S, K, delta_t, T, u, d),
                           desired=leafs)