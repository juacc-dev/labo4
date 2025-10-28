import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylabo.plot as plot
import pylabo.fit as fit
from scipy.special import jv

from errores import error_chi

pd.options.display.float_format = '{:.2e}'.format

RHO = 1.68e-8
RADIUS = 12e-3
MU0 = 4 * np.pi * 1e-7
START = 20
END = None

ERROR_EXTRA = 1

df_v = pd.read_csv("./data/2 - 1-10/con cobre grande A-B.csv")[START:END]
df_i = pd.read_csv("./data/2 - 1-10/con cobre grande I.csv")[START:END]
print(df_v.shape)

freq = df_v["Frecuencia [Hz]"].to_numpy()
volt = np.array(df_v["X [V]"] + 1j * df_v["Y [V]"])
curr = np.array(df_i["X [V]"] + 1j * df_i["Y [V]"])

sens_v = 2 * df_v["Escala [V]"].to_numpy() / 2**16
sens_i = 2 * df_i["Escala [V]"].to_numpy() / 2**16

err_volt = np.sqrt(df_v["Ruido X [V]"].to_numpy() ** 2 + sens_v ** 2) + \
    1j * np.sqrt(df_v["Ruido Y [V]"].to_numpy() ** 2 + sens_v ** 2)

err_curr = np.sqrt(df_i["Ruido X [V]"].to_numpy() ** 2 + sens_i ** 2) + \
    1j * np.sqrt(df_i["Ruido Y [V]"].to_numpy() ** 2 + sens_i ** 2)

phase = np.arctan2(volt.imag, volt.real) - np.arctan2(curr.imag, curr.real)

chi = - np.abs(volt) / (freq * np.abs(curr)) * np.exp(1j * phase)
err_chi = ERROR_EXTRA * error_chi(freq, curr, volt, err_curr, err_volt, phase)

df = pd.DataFrame({
    "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
    "Error Frecuencia [Hz]": pd.Series(),

    "X [V]": chi.real,
    "Error X [V]": np.abs(err_chi.real),

    "Y [V]": chi.imag,
    "Error Y [V]": np.abs(err_chi.imag)
})

df_real = pd.DataFrame({
    "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
    "Error Frecuencia [Hz]": pd.Series(),

    "X [V]": chi.real,
    "Error X [V]": np.abs(err_chi.real),
})

df_imag = pd.DataFrame({
    "Frecuencia [Hz]": df_v["Frecuencia [Hz]"],
    "Error Frecuencia [Hz]": pd.Series(),

    "Y [V]": -chi.imag,
    "Error Y [V]": np.abs(err_chi.imag)
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
p0 = [K0, 0, df["X [V]"].max(), -2.66e-05]

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
    fmt='.',
    ax=[ax[0], ax[1]]
)

plot.datafit(
    df_imag,
    fit_func_imag,
    datalabel="Parte imaginaria",
    fitlabel="Ajuste imaginaria",
    fmt='.',
    ax=[ax[0], ax[2]]
)

ax[0].legend()
ax[1].legend()
ax[2].legend()

plot.show()
