import pandas as pd
import numpy as np
import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11


def split_dataframe(df: pd.DataFrame):
    if df is None:
        return None, None, None, None

    cols = df.columns
    x = df[cols[0]]
    xerr = df[cols[1]]
    y = df[cols[2]]
    yerr = df[cols[3]]

    return x, xerr, y, yerr


def fourier_transform(df: pd.DataFrame, skip_first=0):
    x, _, y, _ = split_dataframe(df)

    sampling_freq = x.size / x.iat[-1]

    yfft = scipy.fft.fft(y)

    cut = y.size // 2

    xf = np.linspace(0, sampling_freq, cut)
    yf = 2.0 / y.size * np.abs(yfft[:cut])

    return xf[skip_first:], yf[skip_first:]


def low_pass_filter(xf, yf, freq):
    cut = np.where(xf < freq)

    return xf[cut], yf[cut]


def find_highest_peaks(x, y, n, separation=0):
    distance = separation / (x[1] - x[0])

    peaks, properties = scipy.signal.find_peaks(
        y,
        distance=distance
    )

    if peaks.size <= n:
        order = np.argsort(-y[peaks])
        return peaks[order]

    heights = y[peaks]

    top_n = np.argpartition(-heights, n - 1)[:n]
    top_n_sorted = top_n[np.argsort(-heights[top_n])]

    highest_peaks = peaks[top_n_sorted]

    return highest_peaks


def plot_data(
    df: pd.DataFrame,
    ax=None,
    **kwargs
):
    x_axis, x_err, y_axis, y_err = split_dataframe(df)

    fig = None

    # ax may be passed. If not, create a new figure
    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
        )

    ax.plot(
        x_axis,
        y_axis,
        **kwargs
    )

    return fig, ax


def plot_fft(xf, yf, peaks, ax=None, **kwargs):
    fig = None

    # ax may be passed. If not, create a new figure
    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
        )

    ax.plot(
        xf,
        yf,
        label="Espectro",
        **kwargs
    )
    ax.plot(
        xf[peaks],
        yf[peaks],
        'x',
        label="Picos"
    )
    for x_pk, y_pk in zip(xf[peaks], yf[peaks]):
        ax.text(x_pk, y_pk, f"{x_pk:.0f} Hz")

    # Línea punteada marcando el pico más bajo
    ax.axhline(
        yf[peaks[-1]],
        linestyle="--",
        color="grey"
    )

    ax.set_yscale("log")
    ax.set(
        xlabel="Frecuencia [Hz]",
        ylabel="Amplitud"
    )
    ax.legend()

    return fig, ax
