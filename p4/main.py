import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import setup_matplotlib, make_dfs, dist
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

# FAKE_ERROR = 0.1
FREQ = 6e-3


def polynomial(x, *a):
    s = 0
    for n, a_n in enumerate(a):
        s += a_n * x ** n

    return s


def make_model(t0, omega):
    def _model(t, kx, ex, *a):
        x_mov = t - t0

        return np.exp(-ex) * np.cos(omega * x_mov - kx) + polynomial(x_mov, *a)

    return _model


# fig, ax = plt.subplots(2, 1)

es = []
es_err = []
kxs = []
kxs_err = []

for ch, df in packs["osc"].items():
    omega = 2 * np.pi * FREQ
    t0 = df["Time"][0]

    model = fit.Function(
        make_model(t0, omega),
        ["kx", "ex", "a0", "a1", "a2"]
    )
    c0 = df["Temperature"].max() - df["Temperature"].min()

    # df["Error Temperature"] = FAKE_ERROR * df["Error Temperature"]

    fit_func = fit.fit(
        model,
        df,
        p0=[0, 1.5, df["Temperature"].mean(), 0, 0]
    )

    df_kx, c, *a = fit_func.param_val
    kx_err, c_err, *a_err = fit_func.param_err

    es.append(c)
    es_err.append(c_err)
    kxs.append(df_kx)
    kxs_err.append(kx_err)

#     plot.datafit(
#         df,
#         fit_func,
#         # datalabel=ch,
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

df_ex = pd.DataFrame({
    "Distancia [mm]": dist,
    "Error dist": 0,
    "epsion x": es,
    "Error ex": es_err
})

df_kx = pd.DataFrame({
    "Distancia [mm]": dist,
    "Error dist": 0,
    "kx": kxs,
    "Error kx": kxs_err
})

fit_func_epsilon = fit.fit(
    fit.funs.linear,
    df_ex
)
fit_func_k = fit.fit(
    fit.funs.linear,
    df_kx
)

epsilon = fit_func_epsilon.param_val[0]
err_epsilon = fit_func_epsilon.param_err[0]
k = fit_func_k.param_val[0]
err_k = fit_func_k.param_err[0]

v = omega / k
err_v = omega * err_k / k ** 2

print(f"epsilon = {epsilon:.2e} +- {err_epsilon:.2e}")
print(f"k = {k:.2e} +- {err_k:.2e}")
print(f"v = {v:.2e} +- {err_v:.2e} mm/s")

print(f"kappa_e = {omega / (2 * epsilon**2):.2f} +- {err_epsilon * omega / epsilon**3:.2f}")
print(f"kappa_v = {v / (2 * omega):.2f} +- {err_v * v / omega:.2f}")
