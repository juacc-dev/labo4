import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylabo.plot as plot
from scipy.special import jv

from util import Material, calculate_chi, radius, rho, try_fit

MU0 = 4 * np.pi * 1e-7

# ERROR_EXTRA = 1

START = 0
END = None


materials = [
    # Material.LATON
    Material.COBRE_1,
    Material.COBRE_GRANDE,
    Material.ALUMINIO_CHICO,
    Material.ALUMINIO_19
]


def model_func(f, k, alpha, scale, offset):
    f = np.array(f)
    x = (-1 + 1j) * np.sqrt(np.pi * MU0 * np.abs(k) * f)

    return - scale * jv(2, x) / jv(0, x) * np.exp(1j * np.pi / 180 * alpha) + offset


params = ["a^2/rho", "alpha", "scale", "offset"]


df_chi = []

for i in range(len(materials)):
    df_v = pd.read_csv(materials[i].value[0])
    df_i = pd.read_csv(materials[i].value[1])

    if df_v.idxmax()["Frecuencia [Hz]"] == 0:
        df_v = df_v[::-1]
        df_i = df_i[::-1]

    df_v = df_v[START:END]
    df_i = df_i[START:END]

    chi, err_chi = calculate_chi(df_v, df_i)

    df = pd.DataFrame({
        "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
        "Error Frecuencia [Hz]": pd.Series(),

        "X [V]": np.real(chi),
        "Error X [V]": np.abs(np.real(err_chi)),

        "Y [V]": -np.imag(chi),
        "Error Y [V]": np.abs(np.imag(err_chi))
    })

    df_chi.append(df)


for i in range(len(materials)):
    name = materials[i].name

    k0 = radius[materials[i]] ** 2 / rho[materials[i]]
    chi_max = (df["X [V]"] ** 2 + df["Y [V]"] ** 2).max()

    p0 = [k0, 0, chi_max, 0]

    df_real, df_imag, fit_func_real, fit_func_imag, n = try_fit(
        df_chi[i],
        model_func,
        params,
        p0=p0
    )

    if df_real is None:
        print(f"Failed to fit {name}")
        fig, ax = plot.data(
            df_chi[i],
            labels=["Chi real", "Chi imaginaria"],
            fmt='.',
        )
        ax.set_title(f"{name} (failed)")
        plot.show()

        continue

    print(f"Material: {name}")
    print("Ajuste real")
    print(fit_func_real.report())
    print("\nAjuste imaginario")
    print(fit_func_imag.report())

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[2, 1]
    )

    plot.datafit(
        df_real,
        fit_func_real,
        datalabel=f"Parte real ({name})",
        fitlabel="Ajuste real",
        reslabel="Parte real",
        fmt='.',
        ax=ax
    )

    plot.datafit(
        df_imag,
        fit_func_imag,
        datalabel=f"Parte imaginaria ({name})",
        fitlabel="Ajuste imaginaria",
        reslabel="Parte imaginaria",
        fmt='.',
        ax=ax
    )

    freq = df_real["Frecuencia [Hz]"]
    y_min = [
        np.min(fit_func_real.f(freq, *fit_func_real.param_val)),
        np.min(fit_func_imag.f(freq, *fit_func_imag.param_val))
    ]
    y_max = [
        np.max(fit_func_real.f(freq, *fit_func_real.param_val)),
        np.max(fit_func_imag.f(freq, *fit_func_imag.param_val))
    ]
    y_min_res = [
        np.min(fit_func_real.residue),
        np.min(fit_func_imag.residue)
    ]
    y_max_res = [
        np.max(fit_func_real.residue),
        np.max(fit_func_imag.residue)
    ]

    # ax[0].set(
    #     ylim=(min(y_min), max(y_max))
    # )
    # ax[1].set(
    #     ylim=(min(y_min_res), max(y_max_res))
    # )

    ax[0].set_title(f"{name} ({n} tries)")
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plot.show()
