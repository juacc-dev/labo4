import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans"
plt.rcParams["font.size"] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams['legend.loc'] = "best"

ERR_NOISE = 0.2 / 100


def value_and_error(df):
    value = np.array(df["X [V]"] + 1j * df["Y [V]"])
    sens = 2 * df["Escala [V]"] / 2 ** 16

    err = np.sqrt(
        df["Ruido X [V]"] ** 2 + sens ** 2 + (value.real * ERR_NOISE) ** 2
    )\
        + 1j * np.sqrt(
        df["Ruido Y [V]"] ** 2 + sens ** 2 + (value.imag * ERR_NOISE) ** 2
    )

    return value, err


def make_df(df_raw):
    z, err = value_and_error(df_raw)

    return pd.DataFrame({
        "Frecuencia [Hz]": df_raw["Frecuencia [Hz]"],
        "Error Frecuencia [Hz]": pd.Series(),
        "X [V]": np.real(z),
        "Error X [V]": np.real(err),
        "Y [V]": np.imag(z),
        "Error Y [V]": np.imag(err),
    })


def complex_values(df):
    return df["X [V]"] + 1j * df["Y [V]"]


def plot(freq, v, v_err, name):
    fig, ax = plt.subplots(2, 2, sharex=True)
    x = np.real(v)
    y = np.imag(v)
    xerr = np.real(v_err)
    yerr = np.imag(v_err)

    ax[0][0].set_title(name)
    ax[0][0].errorbar(freq, x, yerr=xerr, fmt='.', label="Parte real")
    ax[1][0].errorbar(freq, y, yerr=yerr, fmt='.', label="Parte imaginaria")

    abs_err = np.sqrt(((x * xerr) ** 2 + (y * yerr) ** 2) / (x ** 2 + y ** 2))
    phase = np.arctan2(np.imag(v), np.real(v))

    ax[0][1].errorbar(freq, np.abs(v), yerr=abs_err, fmt='.', label="Módulo")
    ax[1][1].errorbar(freq, phase, fmt='.', label="Fase")

    for axis in ax:
        axis[0].legend()
        axis[1].legend()

    plt.show()


def transferencia(df_in, df_out):
    freq = df_in["Frecuencia [Hz]"]

    v_in, err_in = value_and_error(df_in)
    v_out, err_out = value_and_error(df_out)

    t = np.abs(v_out) / np.abs(v_in)

    delta_va = 2 * np.sqrt(
        (np.real(v_out) * np.real(err_out)) ** 2 *
        (np.imag(v_out) * np.imag(err_out)) ** 2
    )

    delta_vb = 2 * np.sqrt(
        (np.real(v_in) * np.real(err_in)) ** 2 *
        (np.imag(v_in) * np.imag(err_in)) ** 2
    )

    terr = t * np.sqrt(
        (delta_va / np.abs(v_out)) ** 2 + (delta_vb / np.abs(v_in)) ** 2
    )

    return pd.DataFrame({
        "Frecuencia [Hz]": freq,
        "Error Frecuencia [Hz]": pd.Series(),
        "Transferencia": t,
        "Error": terr
    })


def even_taylor_cut(x, x0, *a):
    u = (x - x0) ** 2
    s = 0

    for n in range(len(a)):
        s += a[n] * u ** n

    return s
