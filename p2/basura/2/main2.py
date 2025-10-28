import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylabo.plot as plot
import pylabo.fit as fit
from scipy.special import jv

from util import Material, calculate_chi, value_and_error

MU0 = 4 * np.pi * 1e-7

# ERROR_EXTRA = 1

START = 23
END = None


materials = [Material.ALUMINIO_CHICO, Material.ALUMINIO_19]


def model_func(f, k, alpha, scale, offset):
    f = np.array(f)
    x = (-1 + 1j) * np.sqrt(np.pi * MU0 * np.abs(k) * f)

    return - scale * jv(2, x) / jv(0, x) * np.exp(1j * np.pi / 180 * alpha) + offset


def complex_fit(df, name="", k0=0):
    df_real = pd.DataFrame({
        "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
        "Error Frecuencia [Hz]": pd.Series(),

        "X [V]": np.real(chi),
        "Error X [V]": np.abs(np.real(err_chi)),
    })

    df_imag = pd.DataFrame({
        "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
        "Error Frecuencia [Hz]": pd.Series(),

        "Y [V]": -np.imag(chi),
        "Error Y [V]": np.abs(np.imag(err_chi))
    })

    model_params = ["a^2/rho", "alpha", "scale", "offset"]

    model_real = fit.Function(
        lambda f, k, alpha, scale, offset: np.real(
            model_func(f, k, alpha, scale, offset)),
        model_params
    )

    model_imag = fit.Function(
        lambda f, k, alpha, scale, offset: np.imag(
            model_func(f, k, alpha, scale, offset)),
        model_params
    )

    # K0 = RADIUS ** 2 / RHO
    # K0 = 5.88e+03
    chi_max = (df["X [V]"] ** 2 + df["Y [V]"] ** 2).max()
    p0 = [k0, 0, chi_max, 0]

    fit_func_real = fit.fit(
        model_real,
        df_real,
        p0=p0
    )

    fit_func_imag = fit.fit(
        model_imag,
        df_imag,
        p0=p0
    )
    if fit_func_real is None or fit_func_imag is None:
        failed_part = "real" if fit_func_real is None else "imag"
        print(f"Failed to fit {name} ({failed_part}).")
        fit_func_real = None
        fit_func_imag = None

    return df_real, df_imag, fit_func_real, fit_func_imag


df_chi = []
# err_chi = []

for i in range(len(materials)):
    df_v = pd.read_csv(materials[i].value[0])
    df_i = pd.read_csv(materials[i].value[1])

    if df_v["Frecuencia [Hz]"][0] == 2000.0:
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


# fig, ax = plt.subplots(len(materials), 1, sharex=True)

# for i in range(len(materials)):
#     ax[i].set_title(materials[i].name)
#     plot.data(
#         df_chi[i],
#         labels=["Chi real", "Chi imaginaria"],
#         fmt='.',
#         ax=ax[i]
#     )

#     ax[0].legend()

# plot.show()

fig, ax = plt.subplots(
    2 * len(materials),
    1,
    sharex=True,
    height_ratios=[2, 2, 1, 1]
)

print(len(materials))
for i in range(len(materials)):
    name = materials[i].name

    k0 = radius[materials[i]] ** 2 / rho[materials[i]]

    df_real, df_imag, fit_func_real, fit_func_imag = complex_fit(
        df_chi[i],
        k0=k0,
        name=name
    )

    print(name)
    plot.datafit(
        df_real,
        fit_func_real,
        datalabel=f"Parte real ({name})",
        fitlabel="Ajuste real",
        reslabel="Parte real",
        fmt='.',
        ax=[ax[i], ax[i+2]]
    )

    plot.datafit(
        df_imag,
        fit_func_imag,
        datalabel=f"Parte imaginaria ({name})",
        fitlabel="Ajuste imaginaria",
        reslabel="Parte imaginaria",
        fmt='.',
        ax=[ax[i], ax[i+2]]
    )
    ax[i].legend()
    ax[i+2].legend()

plt.tight_layout()
plot.show()
