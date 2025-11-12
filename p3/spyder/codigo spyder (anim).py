import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
import bisect
import time
from pathlib import Path
import logging
from enum import Enum
from matplotlib.animation import FuncAnimation


plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

PATH = Path(
    r"C:\Users\publico\Documents\L4 Grupo 6 2025-2\datos\3 - 29-10"
)

# %%

count = 0

# %%

logger = logging.getLogger("visa")


def list_instruments():
    rm = pyvisa.ResourceManager()

    ids = rm.list_resources()
    for id in ids:
        print(id)


class VisaInstrument:
    def __init__(
        self,
        address,
        **kwargs
    ) -> None:

        self._instrument = pyvisa.ResourceManager().open_resource(
            resource_name=address,
            # read_termination='\n',
            # write_termination='\n',
            **kwargs
        )

        self.id = self.query("*IDN?")

    def __del__(self):
        self._instrument.close()

    def check(self) -> bool:
        id = self.query("*IDN?")

        if id is None:
            self._instrument.close()
            logger.error("Failed to identify intrument")

            return False

        return True

    def reset(self) -> None:
        self.write("*RST")

    def query(
        self,
        cmd: str,
        **kwargs
    ) -> str:
        """Query a single value from the instrument."""

        return self._instrument.query(cmd, **kwargs)

    def query_values(
        self,
        cmd: str,
        *,
        ascii: bool = False,
        **kwargs
    ):
        """Query multiple values from the instrument. By default, the result is
        sent in binary form, but passing ascii=True makes it use plain text."""

        if not ascii:
            return self._instrument.query_binary_values(cmd, **kwargs)

        else:
            return self._instrument.query_ascii_values(cmd, **kwargs)

    def write(
        self,
        cmd: str,
        **kwargs
    ) -> None:
        """Write a single value to the instrument."""

        return self._instrument.write(cmd, **kwargs)

    def write_values(
        self,
        cmd: str,
        *,
        ascii: bool = False,
        **kwargs
    ):
        """Write multiple values to the instrument. By default, the messsage is
        sent in binary form, but passing ascii=True makes it use plain text."""

        if not ascii:
            return self._instrument.write_binary_values(cmd, **kwargs)

        else:
            return self._instrument.write_ascii_values(cmd, **kwargs)

    # def is_done(self) -> bool:
    #     """
    #     This function should block (it doesn't)
    #     """
    #     return self.query("*OPC?") == "1"


logger = logging.getLogger("visa.fungen")


class Funs(Enum):
    SINE = "SINusoid"
    SQUARE = "SQUare"
    PULSE = "PULSe"
    RAMP = "RAMP"
    NOISE = "PRNoise"
    DC = "DC"
    SINC = "SINC"
    GAUSSIAN = "GAUSsian"
    LORENTZ = "LORentz"
    USER1 = "USER1"
    USER2 = "USER2"
    USER3 = "USER3"
    USER4 = "USER4"


class FunctionGenerator(VisaInstrument):
    def __init__(
        self,
        address,
        **kwargs
    ):
        super().__init__(address, **kwargs)

    def output(
        self,
        state,
        ch=1,
    ):
        """Turn output state on or off. Possible arguments are "ON" (or 1) and
        "OFF" (or 0)."""
        if state is not None:
            self.write(f"OUTPut{ch}:STATe {state}")

    def set(
        self,
        ch=1,
        voltage=None,
        freq=None,
        shape: Funs | str = None,
        impedance=None
    ):
        """Set configuarion for the function generator."""

        if voltage is not None:
            self.write(f"SOURce{ch}:VOLTage {voltage}")

        if freq is not None:
            self.write(f"SOURce{ch}:FREQuency {freq}")

        if shape is not None:
            shape = shape.value if type(shape) is Funs else shape

            self.write(f"SOURce{ch}:FUNCtion:SHAPe {shape}")

        if impedance is not None:
            self.write(f"OUTPut{ch}:IMPedance {impedance}")

    def get_voltage(self):
        return float(self.query(f"SOURce1:VOLTage?"))

    def get_freq(self):
        return float(self.query(f"SOURce1:FREQuency?"))

    def get_shape(self):
        return self.query(f"SOURce1:FUNCtion:SHAPe?")

    def get_impedance(self):
        return self.query(f"OUTPut1:IMPedance?")


# %%


def dict_invert(d: dict):
    return {v: k for k, v in d.items()}


def closest_value(sorted_dict, x):
    """Find key that is closest to x and return the value of the key."""

    sorted_list = list(sorted_dict.keys())

    idx = bisect.bisect_left(sorted_list, x)

    if idx == 0:
        return sorted_list[0]
    if idx == len(sorted_list):
        return sorted_list[-1]

    # Candidates are the element just left of the insertion point
    # and the element at the insertion point
    left = sorted_list[idx - 1]
    right = sorted_list[idx]

    # Choose the closer one (if tie, pick the smaller value)
    value = left if abs(x - left) <= abs(right - x) else right

    return sorted_dict[value]


class SR830:
    # Basic methods

    def __init__(self, addr):
        self._instr = pyvisa.ResourceManager().open_resource(addr)

    def __del__(self):
        return self._instr.close()

    def write(self, cmd: str):
        return self._instr.write(cmd)

    def read(self, cmd: str):
        return self._instr.read(cmd)

    def query(self, cmd: str):
        return self._instr.query(cmd)

    def query_values(
        self,
        cmd: str,
        ascii=True,
        separator=",",
        **kwargs
    ):
        if ascii:
            return self._instr.query_ascii_values(cmd, separator=",", **kwargs)

        else:
            return self._instr.query_binary_values(cmd, **kwargs)

# Data
# Precalculated dictionaries and their inverses

    bits = 16
    freq_min = 0.001
    freq_max = 102000.0

    # Full-scale sensitivity
    scale = {  # volts
        0: 2e-09, 1: 5e-09, 2: 1e-08, 3: 2e-08, 4: 5e-08, 5: 1e-07,
        6: 2e-07, 7: 5e-07, 8: 1e-06, 9: 2e-06, 10: 5e-06, 11: 1e-05,
        12: 2e-05, 13: 5e-05, 14: 1e-04, 15: 2e-04, 16: 5e-04, 17: 1e-03,
        18: 2e-03, 19: 5e-03, 20: 1e-02, 21: 2e-02, 22: 5e-02, 23: 1e-01,
        24: 2e-01, 25: 5e-01, 26: 1e+00,
    }
    scale_inv = dict_invert(scale)

    filter_time = {  # seconds
        0: 1e-06, 1: 3e-06, 2: 1e-05, 3: 3e-05, 4: 1e-04, 5: 3e-04,
        6: 1e-03, 7: 3e-03, 8: 1e-02, 9: 3e-02, 10: 1e-01, 11: 3e-01,
        12: 1e+00, 13: 3e+00, 14: 1e+01, 15: 3e+01, 16: 1e+02, 17: 3e+02,
        18: 1e+03, 19: 3e+03,
    }
    filter_time_inv = dict_invert(filter_time)

    filter_slope = {  # dB / oct
        0: 6.0, 1: 12.0, 2: 24.0,
    }
    filter_slope_inv = dict_invert(filter_slope)

    channels = {
        0: "A",
        1: "A-B",
        2: "I",  # 1M
        3: "Ihi",  # 100M
    }
    channels_inv = dict_invert(channels)

    # Setters

    def set_input(self, channel: str):
        """Selecciona el modo de medición
        `mode` puede ser:
          - "A"
          - "A-B"
          - "I"
          - "Ihi": (10M)
          """

        if channel in SR830.channels_inv:
            self.write(f"ISRC {SR830.channels_inv[channel]}")

    def set_scale(self, voltage):
        i = closest_value(SR830.scale_inv, voltage)

        self.write(f"SENS {i}")

        return SR830.scale[i]

    def set_filter_cfg(
        self,
        time=None,   # tiempo de integración del filtro
        slope=None   # pendiente del filtro x6dB/oct
    ):
        ret = []

        if time is not None:
            t = closest_value(SR830.filter_time_inv, time)
            self.write(f"OFLT {t}")

            ret.append(SR830.filter_time[t])

        if slope is not None:
            s = closest_value(SR830.filter_slope_inv, slope)
            self.write(f"OFLS {s}")

            ret.append(SR830.filter_slope[s])

        return ret

    def set_use_internal_ref(self, use: bool):
        if use:
            self.write("FMOD 1")
        else:
            self.write("FMOD 0")

    def set_internal_ref_cfg(
        self,
        freq=None,
        voltage=None,
        use_internal: bool = None
    ):
        ret = []
        if freq is not None:
            self.write(f"FREQ {freq}")
            ret.append(float(self.query("FREQ?")))

        if voltage is not None:
            self.write(f"SLVL {voltage}")
            ret.append(float(self.query("SLVL?")))

        return ret

    def set_display(self, mode):
        """`mode` puede ser:
          - "XY"
          - "RT" (R, theta)
        """

        if mode == "XY":
            self.write("DDEF 1, 0")
            self.write("DDEF 2, 0")

        elif mode == "RT":
            self.write("DDEF 1, 1")
            self.write("DDEF 2, 1")

        else:
            print("Error: modo de diplay inválido.")
            raise Exception

    def set_aux_out(self, aux_out=1, aux_v=0):
        self.write(f"AUXV {aux_out}, {aux_v}")

    # Getters

    # Usaba query_ascii_values...
    def get_scale(self):
        i = int(self.query("SENS?"))

        return SR830.scale[i]

    def get_filter_cfg(self):
        """Returns time_constant, low_pass_slope"""

        t = int(self.query("OFLT?"))
        s = int(self.query("OFLS?"))

        return SR830.filter_time[t], SR830.filter_slope[s]

    def get_display(self):
        return self.query_values("SNAP? 10, 11")

    # Snap (measure)

    def snap(self):
        """X, Y values."""
        return self.query_values("SNAP? 1, 2")

    def snap_rt(self):
        """R, theta values."""
        return self.query_values("SNAP? 3, 4")

    def snap_noise(self):
        x_noise = float(self.query("OUTR? 1"))
        y_noise = float(self.query("OUTR? 2"))

        return complex(x_noise, y_noise)

    def value(self) -> complex:
        """Medición compleja. Devuelve un único número complejo."""

        x, y = self.snap()
        return complex(x, y)

    def right_value(self):
        """Ajusta la escala para medir correctamente."""

        scale = self.get_scale()

        done = False
        while not done:
            x, y = self.snap()
            s = max(abs(x), abs(y))

            # Primera escala mayor que `s`.
            maybe_idx = bisect.bisect_right(list(SR830.scale_inv.keys()), s)

            if maybe_idx == 27:
                maybe_idx = 26

            maybe_scale = SR830.scale[maybe_idx]

            # Si la escala propuesta es la adecuada
            if scale == maybe_scale:
                break

            # Si solo corrige uno para abajo, debe estar bien
            idx_diff = maybe_idx - SR830.scale_inv[scale]
            if idx_diff == -1:
                done = True

            # if maybe_idx > scale.keys()[-1]

            # Si `maybe_scale` es "mucho" menor que `scale`, la elección de
            # escala podría no ser la correcta (por falta de sensibilidad para
            # representar bien a `s`).

            # Prueba con la escala propuesta
            scale = self.set_scale(maybe_scale)

        return complex(x, y)

    # Others

    def save_state(self, buffer=1):
        """`buffer` is an integer between 1 and 9."""

        self.write(f"SSET {buffer}")

    def load_state(self, buffer=1):
        """`buffer` is an integer between 1 and 9."""

        self.write(f"RSET {buffer}")

# %%


def value_and_error(df):
    value = df["X [V]"] + 1j * df["Y [V]"]

    sens = 2 * df["Escala [V]"] / 2 ** 16

    err = np.sqrt(df["Ruido X [V]"] ** 2 + sens ** 2 + (np.real(value) * 0.2 / 100) ** 2) + \
        1j * np.sqrt(df["Ruido Y [V]"] ** 2 + sens ** 2 +
                     (np.imag(value) * 0.2 / 100) ** 2)

    return value, err


def lockin_error(value, noise, scale):
    sens = 2 * scale / 2 ** 16

    err = np.sqrt(np.real(noise) ** 2 + sens ** 2 + (np.real(value) * 0.2 / 100) ** 2) + \
        1j * np.sqrt(np.imag(noise) ** 2 + sens ** 2 +
                     (np.imag(value) * 0.2 / 100) ** 2)

    return value, err


def promedio(lockin, n=30, tiempo=0):
    z = np.zeros(n, dtype=complex)

    for i in range(n):
        z[i] = lockin.right_value()
        time.sleep(tiempo / n)

    z_avg = np.average(z)
    stdev = np.std(np.real(z)) + 1j * np.std(np.imag(z))

    return z_avg, stdev


def measure_point(
    lockin: SR830,
    fungen: FunctionGenerator,
    set_freq,
    n_tau=5,
    q=20
):
    fungen.set(freq=set_freq)
    freq = fungen.get_freq()

    # Mantiene el factor de calidad ~constante
    integration_time = q / freq
    integration_time = lockin.set_filter_cfg(time=integration_time)[0]

    time.sleep((n_tau - 1) * integration_time)

    data, noise = promedio(lockin, n=30, tiempo=1)

    # Report
    print(f"{freq:.1f} Hz: {data.real:.1e} +- {noise.real:.1e} (real) {
          data.imag:.1e} +- {noise.imag:.1e} (imag)")

    scale = lockin.get_scale()

    return freq, data, noise, integration_time, scale


lines = [
    ax[0][0].plot([], [], '.', label="Parte real")[0],
    ax[1][0].plot([], [], '.', label="Parte imaginaria")[0],
    ax[0][1].plot([], [], '.', label="Módulo")[0],
    ax[1][1].plot([], [], '.', label="Fase")[0],
]


def anim_init(ax, lines, freq):
    xlim = (np.min(freq / 1000), np.max(freq / 1000))

    ax[0][0].set(
        ylabel="Amplitud [V]",
        xlim=xlim
    )
    ax[1][0].set(
        xlabel="Frecuencia [kHz]",
        ylabel="Amplitud [V]",
        xlim=xlim
    )
    ax[0][1].set(
        ylabel="Amplitud [V]",
        yscale="log",
        xlim=xlim
    )
    ax[1][1].set(
        xlabel="Frecuencia [kHz]",
        ylabel="Fase [rad]",
        xlim=xlim
    )

    for axis in np.flatten(ax):
        axis.legend()

    return lines


def animate(
    n,
    ax,
    lines,
    lockin: SR830,
    fungen: FunctionGenerator,
    freqs,
    df: pd.DataFrame,
    n_tau=5,
    q=20
):
    freq, data, noise, tau, scale = measure_point(
        lockin,
        fungen,
        freqs[n],
        n_tau=n_tau,
        q=q
    )

    df["X [V]"][n] = np.real(data)
    df["Ruido X [V]"][n] = np.real(noise)

    df["Y [V]"][n] = np.imag(data)
    df["Ruido Y [V]"][n] = np.imag(noise)

    df["Tiempo de integración [s]"][n] = tau
    df["Escala [V]"][n] = scale

    freq_khz = freq / 1000
    v = data
    # v_err = lockin_error(v, noise, scale)

    x = np.real(v)
    # xerr = np.real(v_err)
    y = np.imag(data)
    # yerr = np.imag(v_err)

    abs = np.abs(v)
    phase = np.arctan2(y, x)

    # abs_err = np.sqrt(((x * xerr) ** 2 + (y * yerr) ** 2) / (x ** 2 + y ** 2))
    # phase_err = np.sqrt((xerr / (x ** 2 + 1)) ** 2 +
    #                     (yerr / (y ** 2 + 1)) ** 2)

    lines[0].set_data(freq_khz, x)
    lines[1].set_data(freq_khz, y)
    lines[2].set_data(freq_khz, abs)
    lines[3].set_data(freq_khz, phase)

    for axis in np.flatten(ax):
        axis.set_ylim(df["Y [V]"].min(), df["Y [V]"].max())
    pass


def measure(
    lockin: SR830,
    fungen: FunctionGenerator,
    freqs,
    N_TAU=5,
    Q=20
) -> pd.DataFrame:
    zeros = np.zeros_like(freqs)

    df = pd.DataFrame({
        "Frecuencia [Hz]": zeros,
        "X [V]": zeros,
        "Ruido X [V]": zeros,
        "Y [V]": zeros,
        "Ruido Y [V]": zeros,
        "Tiempo de integración [s]": zeros,
        "Escala [V]": zeros,
    })

    fig, ax = plt.subplots(2, 2, sharex=True)

    anim = FuncAnimation(
        fig,
        partial(animate(ax=ax, lines=lines))
    )

    return df


def transferencia(a, a_err, b, b_err):
    transf = np.abs(a) / np.abs(b)

    delta_va = 2 * np.sqrt(
        (np.real(a) * np.real(a_err)) ** 2 * (np.imag(a) * np.imag(a_err)) ** 2
    )

    delta_vb = 2 * np.sqrt(
        (np.real(b) * np.real(b_err)) ** 2 * (np.imag(b) * np.imag(b_err)) ** 2
    )

    transf_err = transf * np.sqrt(
        (delta_va / np.abs(a)) ** 2 + (delta_vb / np.abs(b)) ** 2
    )

    return transf, transf_err


def plot_measure(df, title=None):
    fig, ax = plt.subplots(
        2, 2,
        sharex=True
    )

    freq = df["Frecuencia [Hz]"] / 1000
    v, v_err = value_and_error(df)

    x = np.real(v)
    xerr = np.real(v_err)
    y = np.imag(v)
    yerr = np.imag(v_err)

    abs = np.abs(v)
    phase = np.arctan2(y, x)

    abs_err = np.sqrt(((x * xerr) ** 2 + (y * yerr) ** 2) / (x ** 2 + y ** 2))
    phase_err = np.sqrt((xerr / (x ** 2 + 1)) ** 2 +
                        (yerr / (y ** 2 + 1)) ** 2)

    ax[0][0].errorbar(freq, x, yerr=xerr,
                      fmt=".", capsize=2, label="Parte real")

    ax[1][0].errorbar(freq, y, yerr=yerr,
                      fmt=".", capsize=2, label="Parte imaginaria")

    ax[0][1].errorbar(freq, abs, yerr=abs_err,
                      fmt=".", capsize=2, label="Módulo")

    ax[1][1].errorbar(freq, phase, yerr=phase_err,
                      fmt=".", capsize=2, label="Fase")

    ax[0][0].set(
        ylabel="Amplitud [V]"
    )
    ax[1][0].set(
        xlabel="Frecuencia [kHz]",
        ylabel="Amplitud [V]"
    )
    ax[0][1].set(
        ylabel="Amplitud [V]"
    )
    ax[1][0].set(
        xlabel="Frecuencia [kHz]",
        ylabel="Fase [rad]"
    )

    for axis in np.flatten(ax):
        axis.legend()
        axis.axvline(freq[np.argmax(abs)], color="black", alpha=0.5)
        axis.axvline(freq[np.argmin(abs)], color="black", alpha=0.5)

    if title is not None:
        ax[0][0].set_title(title)

    plt.show()


def catrange(stops: list[tuple]):
    arrays = []

    # Start at the first stop
    start = stops[0][0]

    # Stop at each stop
    for stop, step in stops[1:]:
        rg = np.arange(start, stop, step)
        arrays.append(rg)

        start = stop

    return np.concatenate(arrays)


# %%

lockin = SR830("GPIB0::7::INSTR")
fungen = FunctionGenerator("USB0::0x0699::0x0346::C033250::INSTR")

print(f"Conectado al Lock-in: {lockin.query('*IDN?')}")
print(f"Conectado al generador: {fungen.query('*IDN?')}")

# %%

Q = 100  # Factor de calidad
FILTER_SLOPE = 12
AMPLITUDE = 1  # volts

# step size
HIGH = 0.1
MED = 1.0
LOW = 10.0
FAR = 100.0

freq = catrange([
    (40e3, None),
    (49e3, FAR),
    (49.9e3, LOW),
    (50e3, LOW),
    (60e3, FAR),
])

lockin.set_use_internal_ref(False)
lockin.set_filter_cfg(slope=FILTER_SLOPE)
lockin.set_scale(1.0)
fungen.set(
    voltage=AMPLITUDE,
    shape=Funs.SINE
)
lockin.set_input("A")

fungen.output("ON")
df = measure(lockin, fungen, freq, Q=Q)
lockin.set_scale(1.0)
fungen.output("OFF")

filepath = PATH / f"({count}) x.csv"
df.to_csv(filepath, index=False)
print(f"Se guardó {filepath.stem}")

count += 1

plot_measure(df, title=filepath.stem)
