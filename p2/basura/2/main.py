import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylabo.plot as plot
import pylabo.fit as fit
from scipy.special import jv

from util import Material, calculate_chi, value_and_error


pd.options.display.float_format = '{:.2e}'.format

# RHO = 1.68e-8  # cobre
RHO = 2.65e-8  # aluminio
RADIUS = 19e-3
MU0 = 4 * np.pi * 1e-7

# ERROR_EXTRA = 1

START = 23
END = None

material = Material.ALUMINIO_CHICO

df_v = pd.read_csv(material.value[0])
df_i = pd.read_csv(material.value[1])

if df_v["Frecuencia [Hz]"][0] == 2000.0:
    df_v = df_v[::-1]
    df_i = df_i[::-1]

df_v = df_v[START:END]
df_i = df_i[START:END]

volt, err_volt = value_and_error(df_v)
curr, err_curr = value_and_error(df_i)
chi, err_chi = calculate_chi(df_v, df_i)

df_volt = pd.DataFrame({
    "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
    "Error Frecuencia [Hz]": pd.Series(),

    "X [V]": np.real(volt),
    "Error X [V]": np.real(err_volt),

    "Y [V]": np.imag(volt),
    "Error Y [V]": np.imag(err_volt),
})

df_curr = pd.DataFrame({
    "Frecuencia [Hz]": df_i["Frecuencia [Hz]"],
    "Error Frecuencia [Hz]": pd.Series(),

    "X [V]": np.real(curr),
    "Error X [V]": np.real(err_curr),

    "Y [V]": np.imag(curr),
    "Error Y [V]": np.imag(err_curr),
})

df_chi = pd.DataFrame({
    "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
    "Error Frecuencia [Hz]": pd.Series(),

    "X [V]": np.real(chi),
    "Error X [V]": np.abs(np.real(err_chi)),

    "Y [V]": -np.imag(chi),
    "Error Y [V]": np.abs(np.imag(err_chi))
})

fig, ax = plt.subplots(3, 1, sharex=True)

plot.data(
    df_volt,
    labels=["Volt real", "Volt imaginaria"],
    fmt='.',
    ax=ax[0]
)
plot.data(
    df_curr,
    labels=["Corriente real", "Corriente imaginaria"],
    fmt='.',
    ax=ax[1]
)
plot.data(
    df_chi,
    labels=["Chi real", "Chi imaginaria"],
    fmt='.',
    ax=ax[2]
)
for axis in ax:
    axis.legend()
plot.show()

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


def model_func(f, k, alpha, scale, offset):
    f = np.array(f)
    x = (-1 + 1j) * np.sqrt(np.pi * MU0 * np.abs(k) * f)

    return - scale * jv(2, x) / jv(0, x) * np.exp(1j * np.pi / 180 * alpha) + offset


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

fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[3, 1, 1])

K0 = RADIUS ** 2 / RHO
K0 = 5.88e+03
CHI_MAX = (df_chi["X [V]"] ** 2 + df_chi["Y [V]"] ** 2).max()
p0 = [K0, 0, CHI_MAX, 0]

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

print(f"Se espera k = {K0:.2e}")
print("Parte real")
print(fit_func_real.report())

print("Parte imaginaria")
print(fit_func_imag.report())

plot.datafit(
    df_real,
    fit_func_real,
    datalabel="Parte real",
    fitlabel="Ajuste real",
    reslabel="Parte real",
    fmt='.',
    ax=[ax[0], ax[1]]
)

plot.datafit(
    df_imag,
    fit_func_imag,
    datalabel="Parte imaginaria",
    fitlabel="Ajuste imaginaria",
    reslabel="Parte imaginaria",
    fmt='.',
    ax=[ax[0], ax[2]]
)

ax[0].legend()
ax[1].legend()
ax[2].legend()

plot.show()
