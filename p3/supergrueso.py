import pandas as pd
import numpy as np
from pathlib import Path
# import pylabo.plot as plot
import matplotlib.pyplot as plt

from util import make_df, value_and_error


file_in = "./data/2 - 22-10/(9) in supergrueso (completo).csv"
file_out = "./data/2 - 22-10/(10) out supergrueso (completo).csv"

df_raw_in = pd.read_csv(file_in)
df_raw_out = pd.read_csv(file_out)

freq = df_raw_in["Frecuencia [Hz]"]

v_in, err_v_in = value_and_error(df_raw_in)
v_out, err_v_out = value_and_error(df_raw_out)


def plot(v, v_err, name):
    fig, ax = plt.subplots(2, 2, sharex=True)
    x = np.real(v)
    y = np.imag(v)
    xerr = np.real(v_err)
    yerr = np.imag(v_err)

    ax[0][0].set_title(name)
    ax[0][0].errorbar(freq, x, yerr=xerr, fmt='.', label="Parte real")
    ax[1][0].errorbar(freq, y, yerr=yerr, fmt='.', label="Parte imaginaria")

    abs_err = np.sqrt(((x * xerr) ** 2 + (y + yerr) ** 2) / (x ** 2 + y ** 2))
    phase = np.arctan2(np.imag(v), np.real(v))

    ax[0][1].errorbar(freq, np.abs(v), yerr=abs_err, fmt='.', label="Módulo")
    ax[1][1].errorbar(freq, phase, fmt='.', label="Fase")

    for axis in ax:
        axis[0].legend()
        axis[1].legend()

    plt.show()


plot(v_in, err_v_in, "Entrada")
plot(v_out, err_v_out, "Salida")
