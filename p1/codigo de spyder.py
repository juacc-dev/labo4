import pandas as pd
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, \
    READ_ALL_AVAILABLE

import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt
from pathlib import Path

# %%

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

PATH = Path(r"C:\Users\publico\Documents\Laboratorio 4, 2025.2 - Grupo 6\datos 10-9")

FREQ_MAX = 2000      # Donde se corta el espectro (en Hz)
SKIP_FIRST = 20      # Para ignorar los primeros puntos del espectro

N_PEAKS = 5          # Número de picos a encontrar
PEAK_SEP = 100       # Separación mínima entre picos (en Hz)

START_ZERO = 50      # Primeros puntos para centrar en cero
ZERO_TOLERANCE = 10  # Qué tan distinto de cero tiene que ser el primer punto
STOP_AFTER = 0       # Hasta qué segundo de la medición usar


# %%

count = 1


# %%

# NI USB-6210 specs

TIMING_ACCURACY = 50e-6    # 50 ppm of sample rate
TIMING_RESOLUTION = 50e-9  # 50 nanoseconds

GAIN_TEMP_COEF = 7.3e-6  # ppm / degree celsius
REF_TEMP_COEF = 5e-6     # ppm / degree celsius
INL_ERROR = 76e-6        # ppm of range

TEMP_CHANGE_INTERNAL = 1
TEMP_CHANGE_EXTERNAL = 10

RESIDUAL_GAIN_ERROR = {
    10:  75e-6,
    5:   85e-6,
    1:   95e-6,
    0.2: 135e-6
}

RESIDUAL_OFFSET_ERROR = {
    10:  20e-6,
    5:   20e-6,
    1:   25e-6,
    0.2: 40e-6
}

OFFSET_TEMP_COEF = {
    10:  34e-6,
    5:   36e-6,
    1:   49e-6,
    0.2: 116e-6
}

COVERAGE_FACTOR = 3  # Number of sigmas to cover

# sigma
RANDOM_NOISE = {
    10:  229e-6,
    5:   118e-6,
    1:   26e-6,
    0.2: 12e-6
}

# This is calculated assuming 1 C (celsius) change from the last internal
# calibration and 10 C change from the last external calibration
GAIN_TEMP_ERROR = GAIN_TEMP_COEF * TEMP_CHANGE_INTERNAL + \
    REF_TEMP_COEF * TEMP_CHANGE_EXTERNAL


def list_devices():
    system = nidaqmx.system.System.local()

    for dev in system.devices:
        print(dev)


def _error_calc(airange):
    gain_error = RESIDUAL_GAIN_ERROR[airange] + GAIN_TEMP_ERROR
    offset_error = RESIDUAL_OFFSET_ERROR[airange] + OFFSET_TEMP_COEF[airange] * TEMP_CHANGE_INTERNAL + INL_ERROR

    n_points = 100
    noise_uncertainty = COVERAGE_FACTOR * RANDOM_NOISE[airange] / np.sqrt(n_points)

    return gain_error, offset_error, noise_uncertainty


def measure(
    channel_name: str,
    duration,            # in seconds
    sampling_freq: int,  # in hertz
    min_val=-10,
    max_val=10
) -> pd.DataFrame:
    n_samples = int(duration * sampling_freq)
    duration = n_samples / sampling_freq

    with nidaqmx.Task() as task:
        # Set analog channel
        ai_channel = task.ai_channels.add_ai_voltage_chan(
            channel_name,
            min_val=min_val,
            max_val=max_val,
            terminal_config=TerminalConfiguration.DIFF
        )

        # Set sampling configuration
        task.timing.cfg_samp_clk_timing(
            sampling_freq,
            samps_per_chan=n_samples,
            sample_mode=AcquisitionType.FINITE
        )

        data = task.read(
            number_of_samples_per_channel=READ_ALL_AVAILABLE,
            timeout=duration+0.1
        )

        airange = ai_channel.ai_max

        # Para checkear
        print(f"Analog input range: {airange}")

    t = np.linspace(0, duration, n_samples)
    y = np.asarray(data)

    # Uncertainty calculation
    t_err = TIMING_ACCURACY / sampling_freq + TIMING_RESOLUTION

    gain_error, offset_error, noise_uncertainty = _error_calc(airange)
    y_err = y * gain_error + airange * offset_error + noise_uncertainty

    df = pd.DataFrame({
        "Tiempo [s]": pd.Series(t),
        "Error X [s]": pd.Series(t_err),
        "Voltaje [V]": pd.Series(y),
        "Error Y [V]": pd.Series(y_err)
    })

    return df

# %%


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

    ax_data.set(
        xlabel="Tiempo [s]",
        ylabel="Amplitud [V]"
    )
    ax_data.legend()

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


# %%

channel_name = "Dev4/ai1"
duration = 5
freq = 10000

df_raw = measure(
    channel_name,
    duration,
    freq
)


zero = np.average(df_raw["Voltaje [V]"][:START_ZERO])
df_raw["Voltaje [V]"] -= zero

start = where_start(df_raw, START_ZERO, ZERO_TOLERANCE)
stop = where_stop(df_raw, start, STOP_AFTER)

# Para plottear después
df_skipped_first = df_raw[:start]
df_skipped_last = df_raw[stop:]

df = df_raw[start:stop]  # Sólo la parte que interesa

xf, yf = fourier_transform(df, skip_first=SKIP_FIRST)
xf, yf = low_pass_filter(xf, yf, FREQ_MAX)

peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)

peaks_df = pd.DataFrame({
    "Frecuencia": peaks
})

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

plot_fft(xf, yf, peaks, ax=ax_fft)

fig.tight_layout()
plt.show()

# %%

name = ""

if name == "":
    name = str(count)
    count += 1

df_raw.to_csv(PATH / f"{name}.csv", index=False)

print(name)
