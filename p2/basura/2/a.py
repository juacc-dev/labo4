import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import value_and_error

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

df_i = pd.read_csv("./data/3 - 8-10/(8) sin nada I.csv")
df_v = pd.read_csv("./data/3 - 8-10/(7) sin nada A-B.csv")

freq = df_i["Frecuencia [Hz]"]
curr, err_curr = value_and_error(df_i)
volt, err_volt = value_and_error(df_v)

plt.errorbar(freq, np.real(curr), yerr=np.real(err_curr), label="IReal")
plt.errorbar(freq, np.imag(curr), yerr=np.imag(err_curr), label="IImaginario")

freq = df_v["Frecuencia [Hz]"]
plt.errorbar(freq, np.real(volt), yerr=np.real(err_volt), label="VReal")
plt.errorbar(freq, np.imag(volt), yerr=np.imag(err_volt), label="VImaginario")
plt.legend()

plt.show()
