import numpy as np
from numpy.testing import assert_allclose
import pytest

import binomial_options_pricing_model.bopm as bopm


@pytest.mark.parametrize("S, K, payoff",
                         [(22, 21, 1), (18, 21, 0)])
def test_call_payoff_scalar(S, K, payoff):
    assert bopm.call_payoff(S, K) == payoff


@pytest.mark.parametrize("S, K, payoff",
                         [(np.array([22, 18]), 21, np.array([1, 0]))])
def test_call_payoff_vector(S, K, payoff):
    n = len(payoff)
    bopm_call_payoff = bopm.call_payoff(S, K)
    assert all(np.allclose(bopm_call_payoff[i], payoff[i]) for i in range(n))


@pytest.mark.parametrize("S, K, payoff",
                         [(22, 21, 0), (18, 21, 3)])
def test_put_payoff_scalar(S, K, payoff):
    assert bopm.put_payoff(S, K) == payoff


@pytest.mark.parametrize("S, K, payoff",
                         [(np.array([22, 18]), 21, np.array([0, 3]))])
def test_put_payoff_vector(S, K, payoff):
    n = len(payoff)
    bopm_call_payoff = bopm.put_payoff(S, K)
    assert all(np.allclose(bopm_call_payoff[i], payoff[i]) for i in range(n))


@pytest.mark.parametrize("r, t, up, down, prob",
                         [(.12, 3/12, 1.1, 0.9, .6523),
                          (.05, 1, 1.2, 0.8, .6282)])
def test_risk_neutral_probability(r, t, up, down, prob):
    return assert_allclose(actual=bopm.risk_neutral_probability(r, t, up, down),
                           desired=prob, rtol=1e-4)


@pytest.mark.parametrize("r, t, up, down",
                         [(.12, 3/12, 1.02, 0.9),
                          (.12, 3/12, 1.1, 1.04)])
def test_risk_neutral_probability_exception(r, t, up, down):
    with pytest.raises(ValueError):
        bopm.risk_neutral_probability(r, t, up, down)


@pytest.mark.parametrize("r, t, up, down, value_up, value_down, value",
                         [(.12, 3/12, 1.1, 0.9, 3.2, 0, 2.0257),
                          (.12, 3/12, 1.1, 0.9, 0, 0, 0),
                          (.12, 3/12, 1.1, 0.9, 2.0257, 0, 1.28)])
def test_expected_payoff(r, t, up, down, value_up, value_down, value):
    p = bopm.risk_neutral_probability(r, t, up, down)
    return assert_allclose(actual=bopm.expected_payoff(r, t, up, down, p, value_up, value_down),
                           desired=value, rtol=1e-2)


@pytest.mark.parametrize("r, t, up, down, p, value_up, value_down, value",
                         [(.12, 3/12, 1.1, 0.9, 0.6523,
                           np.array([3.2, 0, 2.0257]),
                           np.array([0, 0, 0]),
                           np.array([2.0257, 0, 1.28]))])
def test_expected_payoff_vector(r, t, up, down, p, value_up, value_down, value):
    n = len(value)
    bopm_expected_payoff = bopm.expected_payoff(r, t, up, down, p, value_up, value_down)
    assert all(np.allclose(bopm_expected_payoff[i], value[i], rtol=1e-2) for i in range(n))


@pytest.mark.parametrize("S, delta_t, T, up, down, tree",
                         [(20, 3/12, 1/2, 1.1, 0.9,
                           [np.array(20), np.array([18, 22]), np.array([16.2, 19.8, 24.2])]),
                          (50, 1, 2, 1.2, 0.8,
                           [np.array(50), np.array([40, 60]), np.array([32, 48, 72])])
                          ])
def test_price_binomial_tree(S, delta_t, T, up, down, tree):
    n = len(tree)
    bopm_tree = bopm.price_binomial_tree(S, delta_t, T, up, down)
    # Test if each level of the tree is "close enough" (np.allclose).
    assert all(np.allclose(bopm_tree[i], tree[i]) for i in range(n))

####################################################################
####################################################################
# OPTION PRICING


@pytest.mark.parametrize("r, S, K, delta_t, T, up, down, value",
                         [(.12, 20, 21, 3/12, 1/2, 1.1, 0.9, 1.28),
                          (.05, 50, 52, 1, 2, 1.2, 0.8, 7.1416)])
def test_price_european_call(r, S, K, delta_t, T, up, down, value):
    v, _ = bopm.price_option(r, S, K, delta_t, T, up, down, american=False, call=True)
    return assert_allclose(actual=v, desired=value, rtol=1e-2)


@pytest.mark.parametrize("r, S, K, delta_t, T, up, down, value",
                         [(.12, 20, 21, 3/12, 1/2, 1.1, 0.9, 1.28),
                          (.05, 50, 52, 1, 2, 1.2, 0.8, 7.1416)])
def test_price_american_call(r, S, K, delta_t, T, up, down, value):
    v, _ = bopm.price_option(r, S, K, delta_t, T, up, down, american=True, call=True)
    return assert_allclose(actual=v, desired=value, rtol=1e-2)


@pytest.mark.parametrize("r, S, K, delta_t, T, up, down, value",
                         [(.12, 20, 21, 3/12, 1/2, 1.1, 0.9, 1.0591),
                          (.05, 50, 52, 1, 2, 1.2, 0.8, 4.1923)])
def test_price_european_put(r, S, K, delta_t, T, up, down, value):
    v, _ = bopm.price_option(r, S, K, delta_t, T, up, down, american=False, call=False)
    return assert_allclose(actual=v, desired=value, rtol=1e-2)


@pytest.mark.parametrize("r, S, K, delta_t, T, u, d, value",
                         [(.12, 20, 21, 3/12, 1/2, 1.1, 0.9, 1.2686),
                          (.05, 50, 52, 1, 2, 1.2, 0.8, 5.089)])
def test_price_american_put(r, S, K, delta_t, T, u, d, value):
    v, _ = bopm.price_option(r, S, K, delta_t, T, u, d, american=True, call=False)
    return assert_allclose(actual=v, desired=value, rtol=1e-2)


####################################################################
####################################################################
# CCR METHOD


@pytest.mark.parametrize("r, S, K, delta_t, T, sigma, value",
                         [(.12, 20, 21, 3/12, 1/2, np.log(1.1)/np.sqrt(3/12), 1.28),
                          (.05, 50, 52, 1, 2, np.log(1.2)/np.sqrt(1), 4.1923)])
def test_crr_european_put(r, S, K, delta_t, T, sigma, value):
    v, _ = bopm.crr_price_option(r, S, K, delta_t, T, sigma, american=False, call=False)
    return assert_allclose(actual=v, desired=value, atol=.75)


####################################################################
####################################################################
# DELTA HEDGING

@pytest.mark.parametrize("r, S, t, up, down, value_up, value_down, hedge",
                         [(.12, 22, 3/12, 1.1, 0.9, 3.2, 0, (0.727, -13.969)),
                          (.12, 18, 3/12, 1.1, 0.9, 0, 0, (0, 0)),
                          (.12, 20, 3/12, 1.1, 0.9, 2.0257, 0, (0.5075, -8.865))])
def test_hedge_node_scalar(r, S, t, up, down, value_up, value_down, hedge):
    h = bopm.hedge_node(r, S, t, up, down, value_up, value_down)
    return assert_allclose(actual=h, desired=hedge, rtol=1e-2)


@pytest.mark.parametrize("r, S, t, up, down, value_up, value_down, hedge",
                         [(.12, np.array([22, 18, 20]), 3/12, 1.1, 0.9,
                           np.array([3.2, 0, 2.0257]),
                           np.array([0, 0, 0]),
                           np.array([(0.727, -13.969), (0, 0), (0.5075, -8.865)]) )]
                         )
def test_hedge_node_vector(r, S, t, up, down, value_up, value_down, hedge):
    h = bopm.hedge_node(r, S, t, up, down, value_up, value_down)
    return assert_allclose(actual=h, desired=hedge, rtol=1e-2)