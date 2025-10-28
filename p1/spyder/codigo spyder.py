import pandas as pd
import numpy as np
import scipy.fft
import scipy.signal
import scipy.stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import logging
# import nidaqmx
# from nidaqmx.constants import AcquisitionType, TerminalConfiguration, \
#     READ_ALL_AVAILABLE

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

#%%

PATH = Path(
    r"C:\Users\publico\Documents\Laboratorio 4, 2025.2 - Grupo 6\datos 10-9"
)

FREQ_MAX = 500       # Donde se corta el espectro (en Hz)
SKIP_SPECTRUM = 0    # Para ignorar los primeros puntos del espectro

N_PEAKS = 5          # Número de picos a encontrar
PEAK_SEP = 100       # Separación mínima entre picos (en Hz)

# Para detectar dónde empieza la vibración
START_ZERO = 50      # Primeros puntos para tomar como cero
ZERO_TOLERANCE = 10  # Qué tan distinto de cero tiene que ser el primer punto
STOP_AFTER = 0       # Hasta qué segundo de la medición usar

count = 0

#%%

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


# def list_devices():
#     system = nidaqmx.system.System.local()

#     for dev in system.devices:
#         print(dev)


# def _error_calc(airange):
#     gain_error = RESIDUAL_GAIN_ERROR[airange] + GAIN_TEMP_ERROR
#     offset_error = RESIDUAL_OFFSET_ERROR[airange] + \
#         OFFSET_TEMP_COEF[airange] * TEMP_CHANGE_INTERNAL + INL_ERROR

#     n_points = 100
#     noise_uncertainty = COVERAGE_FACTOR * \
#         RANDOM_NOISE[airange] / np.sqrt(n_points)

#     return gain_error, offset_error, noise_uncertainty


# def measure(
#     channel_name: str,
#     duration,            # in seconds
#     sampling_freq: int,  # in hertz
#     min_val=-10,
#     max_val=10
# ) -> pd.DataFrame:
#     n_samples = int(duration * sampling_freq)
#     duration = n_samples / sampling_freq

#     with nidaqmx.Task() as task:
#         # Set analog channel
#         ai_channel = task.ai_channels.add_ai_voltage_chan(
#             channel_name,
#             min_val=min_val,
#             max_val=max_val,
#             terminal_config=TerminalConfiguration.DIFF
#         )

#         # Set sampling configuration
#         task.timing.cfg_samp_clk_timing(
#             sampling_freq,
#             samps_per_chan=n_samples,
#             sample_mode=AcquisitionType.FINITE
#         )

#         data = task.read(
#             number_of_samples_per_channel=READ_ALL_AVAILABLE,
#             timeout=duration+0.1
#         )

#         airange = ai_channel.ai_max

#         # Para checkear
#         print(f"Analog input range: {airange}")

#     t = np.linspace(0, duration, n_samples)
#     y = np.asarray(data)

#     # Uncertainty calculation
#     t_err = TIMING_ACCURACY / sampling_freq + TIMING_RESOLUTION

#     gain_error, offset_error, noise_uncertainty = _error_calc(airange)
#     y_err = y * gain_error + airange * offset_error + noise_uncertainty

#     df = pd.DataFrame({
#         "Tiempo [s]": pd.Series(t),
#         "Error X [s]": pd.Series(t_err),
#         "Voltaje [V]": pd.Series(y),
#         "Error Y [V]": pd.Series(y_err)
#     })

#     return df

#%%


logger = logging.getLogger()


def set_if_none(value, default):
    return default if value is None else value


def chi2_r(
    residue,
    yerr,
    n_data,
    n_params
):
    """Reduced chi squared"""

    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return chi2 / degrees_of_freedom


def r2(
    y_data,
    residue
):
    """R squared"""
    return 1 - np.var(residue) / np.var(y_data)


def p_value(
    residue,
    yerr,
    n_data,
    n_params
):
    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return scipy.stats.chi2.sf(chi2, df=degrees_of_freedom)


def split_params(n, x, *args):
    """Function for internal use. It's common functionality for operator
    overloading."""

    return args[:n], args[n:]


class Function:
    """
    Mathematical function with information about the parameters.
    """

    def __init__(
        self,
        f,               # Callable
        param_str: list[str],  # Parameter names
        eq: str = None      # LaTeX formula
    ):
        self.f = f
        self.param_str = param_str
        self.eq = eq

    def __add__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) + other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    def __sub__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) - other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    def __mul__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) * other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    def __truediv__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(x, *args1) / other.f(x, *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )

    # This is function composition: f & g -> f(g(x))
    def __and__(self, other):
        def f(x, *args):
            n = len(self.param_str)
            args1, args2 = split_params(n, x, *args)

            return self.f(other.f(x, *args2), *args2)

        return Function(
            f,
            self.param_str + other.param_str
        )


class FittedFunction(Function):
    """
    Function class together with numeric parameters and information about the
    fit.
    """

    def __init__(
        self,
        func: Function,
        param_val: list[float],  # Optimal parameters
        param_cov: list[float],  # Covariance matrix
        param_err: list[float],  # Parameter uncertainty
        residue,                 # y_data - y_fit
        tests=None               # table with chi squared and stuff
    ):
        super().__init__(
            func.f,
            func.param_str,
            func.eq
        )

        self.param_val = param_val
        self.param_cov = param_cov
        self.param_err = param_err
        self.residue = residue
        self.tests = tests

    def report(self) -> pd.DataFrame:
        """Create a dataframe with the results of the fit: tests (like reduced
        chi squared) and optimal parameters, the latter with their uncertainty.
        """

        # 1st column: parameter names
        names = list(self.tests.keys()) + self.param_str

        # 2nd column: values / optimal values
        values = list(self.tests.values()) + list(self.param_val)

        # 3rd column: uncertainty. Tests don't have any
        errors = [None for _ in range(
            len(self.tests))] + list(self.param_err)

        df = pd.DataFrame({
            "Parámetro": pd.Series(names),
            "Valor": pd.Series(values),
            "Error": pd.Series(errors)
        })

        return df


def fit_real(
    func: Function,
    data_x,
    data_y,
    p0=None,
    yerr=None,
    absolute_sigma=True
):
    """
    Fit a real function (wraped in the Function class) to data using curve_fit.
    """

    if p0 is None:
        logger.warning(
            "Passing no initial parameters to nonlinear function."
        )

    try:
        param_opt, param_cov = curve_fit(
            func.f,
            data_x,
            data_y,
            p0=p0,
            sigma=yerr,
            absolute_sigma=absolute_sigma
        )

    except RuntimeError as e:
        logger.error(f"Failed to fit function. Error: {e}")
        return None, None

    return param_opt, param_cov


def fit(
    model: Function,
    df: pd.DataFrame,
    p0=None,
    absolute_sigma=True
) -> FittedFunction:
    """
    Fit a function to data.
    Returns an object containing all information about the result.
    """

    x_data, _, y_data, yerr = split_dataframe(df)

    yerr = yerr if not yerr.isna().all() else None

    p_opt, p_cov = fit_real(
        model,
        x_data,
        y_data,
        p0=p0,
        yerr=yerr,
        absolute_sigma=absolute_sigma
    )

    if p_opt is None and p_cov is None:
        return None

    # Error in parameters
    p_err = np.sqrt(np.diag(p_cov))

    y_fit = model.f(x_data, *p_opt)

    residue = y_fit - y_data

    # Tests
    r_sq = r2(y_data, residue)
    chi = chi2_r(
        residue,
        yerr,
        len(residue),
        len(p_opt)
    ) if yerr is not None else None

    p_val = p_value(residue, yerr, len(residue), len(p_opt))

    # New Function object
    fit_func = FittedFunction(
        model,
        p_opt,
        p_cov,
        p_err,
        residue,
        tests={
            "R2": r_sq,
            "chi2 red": chi,
            "P value": p_val
        }
    )

    return fit_func


def plot_fitted(
    df: pd.DataFrame,
    fit_func: FittedFunction,
    ax=None,
    label=None,
    fmt=None,
    **kwargs
):
    """
    Plot the fitted function. More points are used as to draw a smooth curve.
    """
    x_axis, _, _, _ = split_dataframe(df)

    n_points = int(8 * plt.rcParams["figure.dpi"])

    x_fit = np.linspace(x_axis.min(), x_axis.max(), n_points)
    y_fit = fit_func.f(x_fit, *fit_func.param_val)

    fig = None

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
        )

    ax.plot(
        x_fit,
        y_fit,
        label=label
    )

    return fig, ax


def plot_residue(
    df: pd.DataFrame,
    fit_func: FittedFunction,
    ax=None,
    fmt=None,
    ylabel=None,
    **kwargs
):
    """Plot the residue from a fit."""

    x_axis, x_err, _, y_err = split_dataframe(df)

    fig = None

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
        )

    # fmt may be 'o' or '.' depending on the number of points
    ylabel = set_if_none(ylabel, "Residuos")

    # If there is no uncertainty in X, don't plot it
    x_err = x_err if not x_err.isna().all() else None

    ax.axhline(
        y=0,
        color="black",
        alpha=0.9
    )

    ax.errorbar(
        x_axis,
        fit_func.residue,
        xerr=x_err,
        yerr=y_err,
        fmt='.',
        **kwargs
    )

    return fig, ax

#%%


def split_dataframe(df: pd.DataFrame):
    """Para separar un dataframe de datos en sus columnas."""

    cols = df.columns
    x = df[cols[0]]
    xerr = df[cols[1]]
    y = df[cols[2]]
    yerr = df[cols[3]]

    return x, xerr, y, yerr


def where_start(df: pd.DataFrame, start_avg, tolerance, start_extra):
    y = df["Voltaje [V]"].to_numpy()

    sec_to_pt = 1 / float(df["Tiempo [s]"].iat[1] - df["Tiempo [s]"].iat[0])

    near_zero = np.abs(y[:start_avg]).max()
    threshold = near_zero * tolerance
    mask = np.abs(y) > threshold

    first_index = np.flatnonzero(mask)

    if first_index.size != 0:
        return int(first_index[0] + start_extra * sec_to_pt)
    else:
        return None


def where_stop(df, start, stop_after):
    dt = float(df["Tiempo [s]"].iat[1] - df["Tiempo [s]"][0])

    if stop_after != 0:
        return int(start + stop_after / dt)
    else:
        return None


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

        ax.set(
            xlabel="Tiempo [s]",
            ylabel="Amplitud [V]"
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

    model = Function(
        lambda x, a, k, c: a * np.exp(-k * x) + c,
        ["A", "k", "C"]
    )

    fit_func = fit(
        model,
        df_peaks,
        p0=p0
    )

    return df_peaks, fit_func


def plot_decay(
    df,
    df_pos,
    df_neg,
    fit_pos,
    fit_neg
):
    fig, ax = plt.subplots(
        3, 1,
        sharex=True,
        height_ratios=[2, 1, 1]
    )

    plot_data(
        df,
        ax=ax[0],
        label="Datos"
    )

    plot_fitted(
        df_pos,
        fit_pos,
        ax=ax[0],
        label="Ajuste superior"
    )

    plot_fitted(
        df_neg,
        fit_neg,
        ax=ax[0],
        label="Ajuste inferior"
    )

    plot_residue(
        df_pos,
        fit_pos,
        ax=ax[1],
        label="Residuo de ajuste superior"
    )

    plot_residue(
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


#%%

channel_name = "Dev4/ai1"
duration = 5
freq = 10000

# df_raw = measure(
#     channel_name,
#     duration,
#     freq
# )

df_raw = pd.read_csv("../cobre/en cero/1.csv")
df_raw["Erorr Y [V]"] = 0.0015

START_EXTRA = 1

sec_to_pt = 1 / (df_raw["Tiempo [s]"][1] - df_raw["Tiempo [s]"][0])

# start = where_start(df_raw, START_ZERO, ZERO_TOLERANCE, START_EXTRA)
start = int(START_EXTRA * sec_to_pt)

df_skipped = df_raw[:start]
df = df_raw[start:]

xf, yf = fourier_transform(df, skip_first=SKIP_SPECTRUM)
xf, yf = low_pass_filter(xf, yf, FREQ_MAX)

peaks = find_highest_peaks(xf, yf, N_PEAKS, separation=PEAK_SEP)


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
        df_skipped,
        ax=ax_data,
        label="Ignorado",
        color="grey",
        alpha=0.6
    )

ax_data.set(
    xlabel="Tiempo [s]",
    ylabel="Amplitud [V]"
)

plot_fft(xf, yf, peaks, ax=ax_fft)

fig.tight_layout()
plt.show()

#%%

name = ""

if name == "":
    name = str(count)
    count += 1

# df_raw.to_csv(PATH / f"{name}.csv", index=False)

print(name)

#%%


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

plot_decay(
    df,
    df_pos,
    df_neg,
    fit_pos,
    fit_neg
)

plt.show()
