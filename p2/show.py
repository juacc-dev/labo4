import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylabo.plot as plot

from util import Material, calculate_chi, value_and_error

MU0 = 4 * np.pi * 1e-7

# ERROR_EXTRA = 1

START = 20
END = -100

materials = [
    Material.LATON,
]

# materials = Material

for material in materials:
    df_v = pd.read_csv(material.value[0])
    df_i = pd.read_csv(material.value[1])

    if df_v.idxmax()["Frecuencia [Hz]"] == 0:
        df_v = df_v[::-1]
        df_i = df_i[::-1]

    df_v = df_v[START:END]
    df_i = df_i[START:END]

    volt, err_volt = value_and_error(df_v)
    curr, err_curr = value_and_error(df_i)

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

    if curr.size == volt.size:
        chi, err_chi = calculate_chi(df_v, df_i)

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
            df_chi,
            labels=["Chi real", "Chi imaginaria"],
            fmt='.',
            ax=ax[-1]
        )

    else:
        fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].set_title(material.name)

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

    ax[-1].set(
        xlabel=r"Frecuencia [Hz] $V_\text{RMS}$"
    )

    for axis in ax:
        axis.set(
            ylabel="Amplitud [V]"
        )
        axis.legend()

    plot.show()
