import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import \
    fourier_transform, \
    low_pass_filter, \
    find_highest_peaks, \
    plot_data, \
    plot_fft

FREQ_MAX = 2000      # Donde se corta el espectro
SKIP_FIRST = 20      # Para ignorar los primeros puntos del espectro

N_PEAKS = 5          # Número de picos a encontrar
PEAK_SEP = 100       # Separación mínima entre picos, en Hertz

START_ZERO = 50      # Primeros puntos para centrar en cero
ZERO_TOLERANCE = 10  # Qué tan distinto de cero tiene que ser el primer punto
STOP_AFTER = 0       # Hasta qué segundo de la medición usar


def where_start(df: pd.DataFrame, start_avg, tolerance):
    y = df["Voltaje [V]"].to_numpy()

    near_zero = np.abs(y[:start_avg]).max()
    threshold = near_zero * tolerance
    mask = np.abs(y) > threshold

    first_index = np.flatnonzero(mask)

    if first_index.size != 0:
        return int(first_index[0])
    else:
        return None


def where_stop(df, start, stop_after):
    dt = float(df["Tiempo [s]"].iat[1] - df["Tiempo [s]"][0])

    if stop_after != 0:
        return int(start + stop_after / dt)
    else:
        return None


df = pd.read_csv("./data/1.csv")

zero = np.average(df["Voltaje [V]"][:START_ZERO])
df["Voltaje [V]"] -= zero

start = where_start(df, START_ZERO, ZERO_TOLERANCE)
stop = where_stop(df, start, STOP_AFTER)

# Para plottear después
df_skipped_first = df[:start]
df_skipped_last = df[stop:]

# Sólo la parte que interesa
df = df[start:stop]

xf, yf = fourier_transform(df, skip_first=SKIP_FIRST)
xf, yf = low_pass_filter(xf, yf, FREQ_MAX)

peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)

peaks_df = pd.DataFrame({
    "Frecuencia": peaks,
    "Amplitud relativa": yf[peaks] / yf[peaks[0]]
})

print(peaks_df)


# Plot

fig, (ax_data, ax_fft) = plt.subplots(
    2, 1,
    figsize=(9, 7),
    height_ratios=[1, 2]
)

plot_data(
    df,
    ax=ax_data,
    label="Vibración",
)

if start is not None:
    plot_data(
        df_skipped_first,
        ax=ax_data,
        color="grey",
        label="Ignorado"
    )
if stop is not None:
    plot_data(
        df_skipped_last,
        ax=ax_data,
        color="grey"
    )

ax_data.set(
    xlabel="Tiempo [s]",
    ylabel="Amplitud [V]"
)
ax_data.legend()

plot_fft(xf, yf, peaks, ax=ax_fft)

fig.tight_layout()

plt.show()
