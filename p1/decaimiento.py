import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from utils import low_pass_filter, find_highest_peaks, \
    split_dataframe, fourier_transform, plot_data, plot_fft

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

df = pd.read_csv("./cobre/en cero/1.csv")

sec_to_pt = 1 / (df["Tiempo [s]"][1] - df["Tiempo [s]"][0])

df["Erorr Y [V]"] = 0.0015

df = df[int(SKIP_SECS * sec_to_pt):]

xf, yf = fourier_transform(df, skip_first=SKIP_FIRST)
xf, yf = low_pass_filter(xf, yf, FREQ_MAX)

hz_to_pt = 1 / (xf[1] - xf[0])

xf = xf[:int(100 * hz_to_pt)]
yf = yf[:int(100 * hz_to_pt)]

peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)

peaks_df = pd.DataFrame({
    "Frecuencia": peaks
})

# fig, (ax_data, ax_fft) = plt.subplots(
#     2, 1,
#     figsize=(9, 7),
#     height_ratios=[1, 2]
# )

# plot_data(
#     df,
#     ax=ax_data,
#     label="Vibración",
# )

# ax_data.set(
#     xlabel="Tiempo [s]",
#     ylabel="Amplitud [V]"
# )
# ax_data.legend()

# plot_fft(xf, yf, peaks, ax=ax_fft)

# fig.tight_layout()
# plt.show()

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

fig, ax = plt.subplots(
    1, 1
)

plot_data(
    df,
    ax=ax,
    label="Vibración",
)

x, _, y, _ = split_dataframe(df)

ax.plot(
    x.iloc[pos_peaks],
    y.iloc[pos_peaks],
    'x',
    label="Picos",
    color="orange"
)
ax.plot(
    x.iloc[neg_peaks],
    y.iloc[neg_peaks],
    'x',
    label="Picos",
    color="orange"
)

ax.set(
    xlabel="Tiempo [s]",
    ylabel="Amplitud [V]"
)
ax.legend()

fig.tight_layout()
plt.show()
