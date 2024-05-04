"""Plot Cox, Ross and Rubinstein model.

This script contains function(s) to plot function of the price in CRR model.
You can plot multiple subplots of price of any parameter.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_crr_one_dim_slices(r, S, K, delta_t, T, sigma, domain, v_functions):
    figure, axis = plt.subplots(2, 3)

    params = [r, S, K, delta_t, T, sigma]
    x_labels = ["risk-free rate r", "spot price $S_0$", "strike price K",
                "time step $\Delta t$", "maturity T", "volatility $\sigma$"]

    for i in range(len(params)):
        param = params[i]
        v_func = v_functions[i]
        param = np.linspace(domain[0] * param, domain[1] * param)

        V_eu_call = v_func(param, american=False, call=True)
        V_eu_put = v_func(param, american=False, call=False)
        # V_am_call = v_func(param, american=True, call=True) AM CALL = EU CALL
        V_am_put = v_func(param, american=True, call=False)

        axis[i % 2, i % 3].plot(param, V_eu_call, color="blue")
        axis[i % 2, i % 3].plot(param, V_eu_put, color="red")
        # axis[i%2, i%3].plot(param, V_am_call, color="blue")
        axis[i % 2, i % 3].plot(param, V_am_put, color="red", linestyle="dashed")

        axis[i % 2, i % 3].set_xlabel(x_labels[i])
        axis[i % 2, i % 3].set_ylabel("Option value V")
        axis[i % 2, i % 3].legend(["EU=AM call", "EU put", "AM put"])

    plt.show()


if __name__ == "__main__":
    from bopm_one_dim_slices import v_one_dim_slices
    r, S, K, delta_t, T, sigma = .02, 50, 40, 1/12, 2, .3
    domain = (0.8, 1.2)
    v_functions = v_one_dim_slices(r, S, K, delta_t, T, sigma)

    plot_crr_one_dim_slices(r, S, K, delta_t, T, sigma, domain, v_functions)
