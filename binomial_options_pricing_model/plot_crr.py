"""Plot Cox, Ross and Rubinstein model.

This script contains function(s) to plot function of the price in CRR model.
You can plot multiple subplots of price of any parameter.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_crr_price_functions(r, S, K, delta_t, T, sigma, domain, price_functions):
    """Plot price functions of each parameter in CRR model.

    :param r: default risk-free rate of the market.
    :param S: default current underlying asset price.
    :param K: default strike price.
    :param delta_t: default time step (in years).
    :param T: default maturity of an option (in years).
    :param sigma: default volatility of an asset.
    :param price_functions: tuple of price functions of each parameter.
    """
    figure, axis = plt.subplots(2, 3)

    params = r, S, K, delta_t, T, sigma
    x_labels = ["risk-free rate r", "spot price $S_0$", "strike price K",
                "time step $\Delta t$", "maturity T", "volatility $\sigma$"]

    for i in range(len(params)):
        param = params[i]
        v_func = price_functions[i]
        # Domain of the parameter.
        param = np.linspace(domain[0] * param, domain[1] * param)

        # Compute price of vanilla options. (american call == european call)
        V_eu_call = v_func(param, american=False, call=True)
        V_eu_put = v_func(param, american=False, call=False)
        # V_am_call = v_func(param, american=True, call=True)
        V_am_put = v_func(param, american=True, call=False)

        # Plot on 2 x 3 grid.
        axis[i % 2, i % 3].plot(param, V_eu_call, color="blue")
        axis[i % 2, i % 3].plot(param, V_eu_put, color="red")
        # axis[i%2, i%3].plot(param, V_am_call, color="blue")
        axis[i % 2, i % 3].plot(param, V_am_put, color="red", linestyle="dashed")

        # Add inequalities to V_K and V_S.
        # call >= S - Ke^(-rT), put >= Ke^(-rT) - S
        if i == 1:
            axis[i % 2, i % 3].plot(param, param - K * np.exp(-r * T), color="blue", linestyle="dotted")
            axis[i % 2, i % 3].plot(param, K * np.exp(-r * T) - param, color="red", linestyle="dotted")
        elif i == 2:
            axis[i % 2, i % 3].plot(param, S - param * np.exp(-r * T), color="blue", linestyle="dotted")
            axis[i % 2, i % 3].plot(param, param * np.exp(-r * T) - S, color="red", linestyle="dotted")

        axis[i % 2, i % 3].set_xlabel(x_labels[i])
        axis[i % 2, i % 3].set_ylabel("Option value V")
        axis[i % 2, i % 3].legend(["EU=AM call", "EU put", "AM put"])

    plt.show()


if __name__ == "__main__":
    from bopm_crr_price_functions import crr_price_functions

    r, S, K, delta_t, T, sigma = .02, 50, 52, 1 / 12, 2, .3
    domain = (0.8, 1.2)
    price_functions = crr_price_functions(r, S, K, delta_t, T, sigma)

    plot_crr_price_functions(r, S, K, delta_t, T, sigma, domain, price_functions)
