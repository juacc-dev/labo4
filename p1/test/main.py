import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from utils import low_pass_filter, find_highest_peaks, \
    split_dataframe, fourier_transform, plot_fft, peak_fit

import pylabo.fit
import pylabo.plot

FREQ_MAX = 2000      # Donde se corta el espectro (en Hz)
SKIP_FIRST = 0      # Para ignorar los primeros puntos del espectro

N_PEAKS = 5          # Número de picos a encontrar
PEAK_SEP = 100       # Separación mínima entre picos (en Hz)

START_ZERO = 50      # Primeros puntos para centrar en cero
ZERO_TOLERANCE = 10  # Qué tan distinto de cero tiene que ser el primer punto
STOP_AFTER = 0       # Hasta qué segundo de la medición usar

SKIP_SECS = 1


df = pd.read_csv("./datos/datos 10-9/laton/variando L/L=205.csv")

sec_to_pt = 1 / (df["Tiempo [s]"][1] - df["Tiempo [s]"][0])

df["Erorr Y [V]"] = 0.0015

df = df[int(SKIP_SECS * sec_to_pt):]

df_fourier = fourier_transform(df, skip_first=SKIP_FIRST)
xf, _, yf, _ = split_dataframe(df_fourier)
xf, yf = low_pass_filter(xf, yf, FREQ_MAX)

hz_to_pt = 1 / (xf[1] - xf[0])

xf = xf[:int(100 * hz_to_pt)]
yf = yf[:int(100 * hz_to_pt)]

peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)

peaks_df = pd.DataFrame({
    "Frecuencia": peaks
})

delta_t = 1 / xf[peaks[0]]
delta_i = int(delta_t * sec_to_pt * 2)

pos_peaks, _ = scipy.signal.find_peaks(
    df["Voltaje [V]"],
    distance=delta_i
)
neg_peaks, _ = scipy.signal.find_peaks(
    -df["Voltaje [V]"],
    distance=delta_i
)

df_pos, fit_pos = peak_fit(df, pos_peaks)
df_neg, fit_neg = peak_fit(df, neg_peaks, p0=[-1, 1, 1])

print(fit_pos.report())
print(fit_neg.report())


# pylabo.plot.fulfit(df_pos, fit_neg)
# plt.show()

fig, ax = plt.subplots(
    3, 1,
    sharex=True,
    height_ratios=[2, 1, 1]
)

pylabo.plot.data(
    df,
    ax=ax[0],
    label="Datos"
)

pylabo.plot.fitted(
    df_pos,
    fit_pos,
    ax=ax[0],
    label="Ajuste superior"
)

pylabo.plot.fitted(
    df_neg,
    fit_neg,
    ax=ax[0],
    label="Ajuste inferior"
)

pylabo.plot.residue(
    df_pos,
    fit_pos,
    ax=ax[1],
    label="Residuo de ajuste superior"
)

pylabo.plot.residue(
    df_neg,
    fit_pos,
    ax=ax[2],
    label="Residuo de ajuste inferior"
)


for axis in ax:
    axis.set(
        ylabel="Voltaje [V]"
    )
    axis.legend()

ax[2].set(xlabel="Tiempo [s]")

plt.show()
