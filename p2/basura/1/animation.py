import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11


delta_t = 1 / peaks[0]
delta_i = delta_t / (t[1] - t[0])
win = 200 * int(delta_i)

fig, ax = plt.subplots(
    2, 1,
    figsize=(9, 7),
    height_ratios=[1, 2]
)

lines = [
    ax[0].plot([], [], '-', label="Considerado", color="blue")[0],
    ax[0].plot([], [], '-', label="Ignorado", color="grey")[0],
    ax[1].plot([], [], '-', label="Espectro")[0],
    ax[1].plot([], [], 'x', label="Picos")[0],
]
lines.extend([
    ax[1].text(0, 0, "") for _ in range(N_PEAKS)
])


def init():
    ax[0].set(
        xlabel="Tiempo [s]",
        ylabel="Amplitud [V]",
        xlim=(0, t[-1]),
        ylim=(v.min(), v.max())
    )
    ax[1].set(
        xlim=(0, FREQ_MAX),
        ylim=(1e-9, 1e-3),
        xlabel="Frecuencia [Hz]",
        ylabel="Amplitud"
    )
    ax[1].set_yscale("log")
    ax[0].legend()
    ax[1].legend()

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
