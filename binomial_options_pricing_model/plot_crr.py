"""Plot Cox, Ross and Rubinstein model.

This script contains function(s) to plot function of the price in CRR model.
You can plot multiple subplots of price of any parameter.
"""

import matplotlib.pyplot as plt


def plot_crr(r, S, K, delta_t, T, sigma):
    figure, axis = plt.subplots(2, 3)
    figure.legend(["EU call", "EU put", "AM call", "AM put"])

    params = [r, S, K, delta_t, T, sigma]
    v_functions = [v_r, v_s, v_k, v_delta_t, v_t, v_sigma]
    for i, (param, v_func) in enumerate(zip(params, v_functions)):
        param = np.linspace(.8 * param, 1.2 * param)

        V_eu_call = v_func(param, american=False, call=True)
        V_eu_put = v_func(param, american=False, call=False)
        V_am_call = v_func(param, american=True, call=True)
        V_am_put = v_func(param, american=True, call=False)

        axis[i % 2, (i) % 3].plot(param, V_eu_call, color="blue", linestyle="dashed")
        axis[i % 2, (i) % 3].plot(param, V_eu_put, color="red", linestyle="dashed")
        axis[i % 2, (i) % 3].plot(param, V_am_call, color="blue")
        axis[i % 2, (i) % 3].plot(param, V_am_put, color="red")

        axis[i % 2, (i) % 3].set_xlabel("test")
        axis[i % 2, (i) % 3].set_ylabel("Option value V")

    plt.plot()

    plot_crr(r, S, K, delta_t, T, sigma)
    plt.plot()