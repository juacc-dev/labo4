import pandas as pd
import numpy as np
# import nidaqmx
# from nidaqmx.constants import AcquisitionType, TerminalConfiguration, \
#     READ_ALL_AVAILABLE

import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt
# from pathlib import Path


plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11


def split_dataframe(df: pd.DataFrame):
    """Para separar un dataframe de datos en sus columnas."""

    cols = df.columns
    x = df[cols[0]]
    xerr = df[cols[1]]
    y = df[cols[2]]
    yerr = df[cols[3]]

    return x, xerr, y, yerr


def fourier_transform(df, skip_first=0):
    x, _, y, _ = split_dataframe(df)

    sampling_freq = x.size / x.iat[-1]
    cut = y.size // 2

    fft = np.fft.fft(y)
    yfft = 2.0/len(y) * np.abs(fft)

    cut = y.size // 2

    xf = np.arange(0, sampling_freq, 1 / (x.iat[-1] - x.iat[0]))
    yf = 2.0 / y.size * np.abs(yfft[:cut])

    df = pd.DataFrame({
        "Frecuencia": xf,
        "X err": pd.Series(),
        "Amplitud": yf,
        "Y err": pd.Series()
    })

    return xf[skip_first:], yf[skip_first:]


def low_pass_filter(xf, yf, freq):
    """Para recortar el espectro."""

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


def plot_data(df: pd.DataFrame, ax=None, **kwargs):
    """Para plottear un dataframe."""

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

    ax.fill_between(
        x_axis,
        y_axis - y_err,
        y_axis + y_err,
        label="Error",
        color="green",
        alpha=0.5
    )

    return fig, ax


def plot_fft(xf, yf, peaks, ax=None, **kwargs):
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


def peak_fit(
    df,
    peaks,
    p0=[1, 1, 1]
):
    x, _, y, y_err = split_dataframe(df)

    df_peaks = pd.DataFrame({
        "x": x.iloc[peaks],
        "x_err": pd.Series(),
        "y": y.iloc[peaks],
        "y_err": y_err.iloc[peaks],
    })

    model = pylabo.fit.Function(
        lambda x, a, k, c: a * np.exp(-k * x) + c,
        ["A", "k", "C"]
    )

    fit_func = pylabo.fit.fit(
        model,
        df_peaks,
        p0=p0
    )

    return df_peaks, fit_func
