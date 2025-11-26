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

# FAKE_ERROR = 0.2
FREQ = 6e-3
omega = 2 * np.pi * FREQ


def polynomial(x, *a):
    s = 0
    for n, a_n in enumerate(a):
        s += a_n * x ** n

    return s


def make_model(t0, omega):
    def _model(t, kx, ex, *a):
        x_mov = t - t0

        return np.exp(-ex) * np.sin(omega * x_mov - kx) + polynomial(x_mov, *a)

    return _model


def find_fit(df):
    t0 = df["Time"][0]

    model = fit.Function(
        make_model(t0, omega),
        ["kx", "ex", "a0", "a1", "a2"]
    )

    # df["Error Temperature"] = FAKE_ERROR * df["Error Temperature"]

    fit_func = fit.fit(
        model,
        df,
        p0=[0, 0, df["Temperature"].mean(), 0, 0]
    )

    kx, ex, *a = fit_func.param_val
    kx_err, ex_err, *a_err = fit_func.param_err

    return fit_func, ex, ex_err, kx, kx_err


def calc_params(exs, exs_err, kxs, kxs_err, do_plot=True):
    df_ex = pd.DataFrame({
        "Distancia $x$ [mm]": dist,
        "Error dist": 0,
        r"$\epsilon x$": exs,
        "Error ex": exs_err
    })

    df_kx = pd.DataFrame({
        "Distancia $x$ [mm]": dist,
        "Error dist": 0,
        r"$kx$": kxs,
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

    if do_plot:
        fig, ax = plt.subplots(
            2,
            2,
            height_ratios=[2, 1],
            sharex=True
        )
        plot.datafit(
            df_ex,
            fit_func_epsilon,
            ax=[ax[0][0], ax[1][0]],
            force_label=True
        )

        plot.datafit(
            df_kx,
            fit_func_k,
            datalabel="k X",
            ax=[ax[0][1], ax[1][1]],
            force_label=True
        )
        ax[0][0].set_title("Decaimiento")
        ax[0][1].set_title("Num. de onda")
        plt.tight_layout()
        plot.show()

    return epsilon, err_epsilon, k, err_k


fig, ax = plt.subplots(2, 1, height_ratios=[2, 1])

exs = []
exs_err = []
kxs = []
kxs_err = []

for ch, df in packs["osc"].items():
    fit_func, ex, ex_err, kx, kx_err = find_fit(df)

    exs.append(ex)
    exs_err.append(ex_err)
    kxs.append(kx)
    kxs_err.append(kx_err)

    plot.datafit(
        df,
        fit_func,
        datalabel=ch,
        ax=ax,
    )

ax[0].set(
    ylabel="Temperatura [C]"
)
ax[1].set(
    ylabel="Resiuos [C]",
    xlabel="Tiempo [s]"
)
plt.tight_layout()
plot.show()

kxs = np.unwrap(kxs)

epsilon, err_epsilon, k, err_k = calc_params(exs, exs_err, kxs, kxs_err)

v = omega / k
err_v = omega * err_k / k ** 2

# print(f"epsilon = {epsilon:.2e} +- {err_epsilon:.2e}")
# print(f"k = {k:.2e} +- {err_k:.2e}")
print(f"v = {v:.2e} +- {err_v:.2e} mm/s")
print(f"kappa_e = {omega / (2 * epsilon**2):.2f} +- {err_epsilon * omega / epsilon**3:.2f}")
print(f"kappa_v = {v / (2 * omega):.2f} +- {err_v * v / omega:.2f}")
