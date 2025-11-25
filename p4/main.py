import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from util import setup_matplotlib, make_dfs, plot_dfs, dist
import pylabo.plot as plot
import pylabo.fit as fit

setup_matplotlib()

packs = {
    "primer intento":   make_dfs("./data/2 - 19-11/0"),
    "enfriado corto":   make_dfs("./data/2 - 19-11/1"),
    "subida 1":         make_dfs("./data/2 - 19-11/2"),
    "subida 2":         make_dfs("./data/2 - 19-11/3"),
    "osc":              make_dfs("./data/2 - 19-11/4 (oscilando)"),
    "enfriado":         make_dfs("./data/2 - 19-11/5 (enfriado)"),
}

FREQ = 6e-3


# for name, df in packs.items():
#     fig, ax = plt.subplots(1, 1)
#     plot_dfs(df, ax=ax, title=name)
#     plt.show()

def poly(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x ** 2


def make_model(t0, omega):
    def _model(x, delta, c, a0, a1, a2):
        t = x - t0

        return np.abs(c) * np.cos(omega * x + delta) + poly(t, a0, a1, a2)

    return _model


# fig, ax = plt.subplots(2, 1)

ampl = []
ampl_err = []
deltas = []
delta_err = []

for ch, df in packs["osc"].items():
    omega = 2 * np.pi * FREQ

    model = fit.Function(
        make_model(df["Time"][0], omega),
        ["delta", "c", "a0", "a1", "a2"]
    )
    c0 = df["Temperature"].max() - df["Temperature"].min()

    fit_func = fit.fit(
        model,
        df,
        p0=[0, 1.5, df["Temperature"].mean(), 0, 0]
    )

    # plot.datafit(
    #     df,
    #     fit_func,
    #     datalabel=ch
    # )

    # plot.show()
    delta, c, a0, a1, a2 = fit_func.param_val

    t = df["Time"] - df["Time"][0]
    # df["Temperature"] -= c * np.cos(omega * t + delta)
    df["Temperature"] -= poly(t, a0, a1, a2)
    # plot.data(df, label=ch, ax=ax, no_yerr=True)

    model2 = fit.Function(
        lambda x, a, d: a * np.cos(omega * x + d),
        ["a", "d"]
    )

    fit_func2 = fit.fit(
        model2,
        df,
        p0=[c, delta]
    )

    ampl.append(fit_func2.param_val[0])
    ampl_err.append(fit_func2.param_err[0])
    deltas.append(fit_func2.param_val[1])
    delta_err.append(fit_func2.param_err[1])
#     print(ch)
#     print(f"  - a = {fit_func2.param_val[0]:.2e} +- {fit_func2.param_err[0]:.2e}")
#     print(f"  - d = {fit_func2.param_val[1]:.2e} +- {fit_func2.param_err[1]:.2e}\n")

#     plot.datafit(
#         df,
#         fit_func2,
#         datalabel=ch,
#         ax=ax,
#         no_yerr=True
#     )

# ax[0].set(
#     ylabel="Temperatura [C]"
# )
# ax[1].set(
#     ylabel="Resiuos [C]",
#     xlabel="Tiempo [s]"
# )
# plt.tight_layout()
# plot.show()

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].errorbar(dist[:-1], ampl, yerr=ampl_err)
ax[1].errorbar(dist[:-1], deltas, yerr=delta_err)
ax[0].set(
    ylabel="Amplitud [C]"
)
ax[1].set(
    xlabel="Distancia [mm]",
    ylabel="Amplitud [C]"
)

plt.show()
