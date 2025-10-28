import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import low_pass_filter, find_highest_peaks, \
    split_dataframe


FREQ_MAX = 500       # Donde se corta el espectro (en Hz)
SKIP_FIRST = 0       # Para ignorar los primeros puntos del espectro

N_PEAKS = 5          # Número de picos a encontrar
PEAK_SEP = 100       # Separación mínima entre picos (en Hz)

START_ZERO = 50      # Primeros puntos para centrar en cero
ZERO_TOLERANCE = 10  # Qué tan distinto de cero tiene que ser el primer punto
STOP_AFTER = 0       # Hasta qué segundo de la medición usar


df = pd.read_csv("./cobre/en cero/1.csv")


def fft(x, y, skip_first=0):
    sampling_freq = x.size / x[-1]
    cut = y.size // 2

    fft = np.fft.fft(y)
    yfft = 2.0/len(y) * np.abs(fft)

    cut = y.size // 2

    xf = np.arange(0, sampling_freq, 1 / (x[-1] - x[0]))
    yf = 2.0 / y.size * np.abs(yfft[:cut])

    return xf[skip_first:], yf[skip_first:]


fig, (ax_data, ax_fft) = plt.subplots(
    2, 1,
    figsize=(9, 7),
    height_ratios=[1, 2]
)

t, _, v, _ = split_dataframe(df)
t = t.to_numpy()
v = v.to_numpy()

xf, yf = fft(t, v)
xf, yf = low_pass_filter(xf, yf, FREQ_MAX)
peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)

delta_t = 1 / peaks[0]
delta_i = delta_t / (t[1] - t[0])
win = 200 * int(delta_i)


ax_data.plot(t, v, color="grey")

lines = [
    ax_data.plot([], [], '-', label="Considerado", color="blue")[0],
    ax_data.plot([], [], '-', label="Ignorado", color="grey")[0],
    ax_fft.plot([], [], '-', label="Espectro")[0],
    ax_fft.plot([], [], 'x', label="Picos")[0],
]
lines.extend([
    ax_fft.text(0, 0, "") for _ in range(N_PEAKS)
])


def init():
    ax_data.set(
        xlabel="Tiempo [s]",
        ylabel="Amplitud [V]",
        xlim=(0, t[-1]),
        ylim=(v.min(), v.max())
    )
    ax_fft.set(
        xlim=(0, FREQ_MAX),
        ylim=(1e-9, 1e-3),
        xlabel="Frecuencia [Hz]",
        ylabel="Amplitud"
    )
    ax_fft.set_yscale("log")
    ax_data.legend()
    ax_fft.legend()

    return lines


def animate(i):
    x = t[i - win:i + win]
    y = v[i - win:i + win]

    x_old = t[:i - win]
    y_old = v[:i - win]

    xf, yf = fft(x, y, skip_first=SKIP_FIRST)
    xf, yf = low_pass_filter(xf, yf, FREQ_MAX)
    peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)

    lines[0].set_data(x, y)

    lines[1].set_data(x_old, y_old)

    lines[2].set_data(
        xf,
        yf
    )

    lines[3].set_data(
        xf[peaks],
        yf[peaks]
    )
    for n, (x_pk, y_pk) in enumerate(zip(xf[peaks], yf[peaks])):
        lines[n+4].set_x(x_pk)
        lines[n+4].set_y(y_pk)
        lines[n+4].set_text(f"{x_pk:.0f} Hz")

    # print(f"[0F[0J{100 * i / t.size}%", end='\n')

    return lines


anim = FuncAnimation(
    fig,
    animate,
    frames=range(win, t.size - win, 60),
    init_func=init,
    interval=16,
    repeat=False,
    blit=True
)

plt.show()
