import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
import bisect
import time
from scipy.optimize import curve_fit
from scipy.special import jv
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams['legend.loc'] = "best"
plt.rcParams['legend.fontsize'] = 11

PATH = Path(
    r"C:\Users\publico\Documents\L4 2025-2 grupo 6\datos\3 - 8-10"
).mkdir(parents=True, exist_ok=True)


MU0 = 4 * np.pi * 1e-7

RHO = {
    "cobre": 1.68e-8,
    "aluminio": 2.65e-8
}

RADIO = {
    "aluminio chico": 0,
    "cobre chico": 12e-3,
    "cobre grande": 0,
}

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


def promedio(lockin, n=30, tau=0):
    z = np.zeros(n, dtype=complex)

    for i in range(n):
        z[i] = lockin.right_value()
        time.sleep(tau / n)

    z_avg = np.average(z)
    stdev = np.std(np.real(z)) + 1j * np.std(np.imag(z))

    return z_avg, stdev


def measure(lockin: SR830, freqs, N_TAU=5, Q=20) -> pd.DataFrame:
    data = np.zeros_like(freqs, dtype=complex)
    noise = np.zeros_like(freqs, dtype=complex)

    # Error en frecuencia y en amplitud
    times = np.zeros_like(freqs)
    scales = np.zeros_like(freqs)

    for i in range(freqs.size):

        freq = lockin.set_internal_ref_cfg(freq=freqs[i])[0]

        print(f"Frecuencia: {freq:.0f} Hz")
        # Mantiene el factor de calidad ~constante
        integration_time = Q / freq
        integration_time = lockin.set_filter_cfg(time=integration_time)[0]

        time.sleep((N_TAU - 1) * integration_time)

        data[i], noise[i] = promedio(lockin, n=30, tau=integration_time)

        # Creo??
        times[i] = integration_time
        scales[i] = lockin.get_scale()

    # return data
    df = pd.DataFrame({
        "Frecuencia [Hz]": freqs,
        "X [V]": np.real(data),
        "Ruido X [V]": np.real(noise),
        "Y [V]": np.imag(data),
        "Ruido Y [V]": np.imag(noise),
        "Tiempo de integración [s]": times,
        "Escala [V]": scales
    })

    return df


def plot_measure(df):
    fig, ax = plt.subplots(
        1, 1
    )

    freq = df["Frecuencia [Hz]"]
    x = df["X [V]"]
    xerr = np.abs(df["Ruido X [V]"])
    y = df["Y [V]"]
    yerr = np.abs(df["Ruido Y [V]"])

    ax.errorbar(freq, x, yerr=xerr, fmt=".", capsize=2, label="Parte real")
    ax.errorbar(freq, y, yerr=yerr, fmt=".",
                capsize=2, label="Parte imaginaria")

    ax.legend()
    ax.set(
        xlabel="Frecuencia [Hz]",
        ylabel="Amplitud"
    )

    plt.show()


def errores(curr, volt, err_curr, err_volt, alpha):
    Xx, Yx = np.real(curr), np.imag(curr)
    Xy, Yy = np.real(volt), np.imag(volt)
    eXx, eYx = np.real(err_curr), np.imag(err_curr)
    eXy, eYy = np.real(err_volt), np.imag(err_volt)

    ux = np.sqrt(Xx**2 + Yx**2)
    err_ux = np.sqrt((Xx * eXx)**2 + (Yx * eYx)**2)/ux

    uy = np.sqrt(Xy**2 + Yy**2)
    err_uy = np.sqrt((Xy * eXy)**2 + (Yy * eYy)**2)/uy

    err_alfa = np.sqrt((np.sqrt((eYy / Xy)**2+(eXy * Yy / Xy ** 2) ** 2)/(1+(Yy/Xy)**2))
                       ** 2 + (np.sqrt((eYx / Xx)**2 + (eXx * Yx / Xx ** 2) ** 2)/(1 + (Yx / Xx)**2))**2)

    cov_ux_alfa = ((Xx*Yx)*(Xx**2+Yx**2)**(-2/3)) * \
        (eYx**2-eXx**2)  # ( Xx*Yx / ux**3) * (eYx**2- eXx**2)
    cov_uy_alfa = ((Xy*Yy)*(Xy**2+Yy**2)**(-2/3)) * \
        (-eYy**2+eXy**2)  # ( Xy*Yy / uy**3) * (eXy**2- eYy**2)

    return err_alfa, err_ux, err_uy, cov_ux_alfa, cov_uy_alfa


def chi_err_real(uy, err_uy, ux, err_ux, alfa, err_alfa, cov_uy_alfa, cov_ux_alfa, f):
    valor = -(uy/(f*ux))*np.sin(alfa)
    err = valor*np.sqrt((err_uy / uy)**2 + (err_ux / ux)**2 + (err_alfa / np.tan(alfa))
                        ** 2 + (2 / np.tan(alfa))*(cov_uy_alfa / uy - cov_ux_alfa / ux))
    return err


def chi_err_imag(uy, err_uy, ux, err_ux, alfa, err_alfa, cov_uy_alfa, cov_ux_alfa, f):
    valor = -(uy/(f*ux))*np.cos(alfa)
    err = valor*np.sqrt((err_uy / uy)**2 + (err_ux / ux)**2 + (err_alfa * np.tan(alfa))
                        ** 2 + (2 * (-np.tan(alfa))) * (cov_uy_alfa / uy - cov_ux_alfa / ux))
    return err


def error_chi(freq, curr, volt, err_curr, err_volt, phase):
    err_phase, err_curr_abs, err_volt_abs, cov_curr, cov_volt = errores(
        curr, volt, err_curr, err_volt, phase
    )

    err_real = chi_err_real(
        np.abs(volt), err_volt_abs, np.abs(
            curr), err_curr_abs, phase, err_phase,
        cov_volt, cov_curr, freq
    )

    err_imag = chi_err_imag(
        np.abs(volt), err_volt_abs, np.abs(
            curr), err_curr_abs, phase, err_phase,
        cov_volt, cov_curr, freq
    )

    return err_real + 1j * err_imag


def calculate_chi(df_v, df_i):
    freq = df_v["Frecuencia [Hz]"]

    # Voltaje y corriente como números complejos
    volt = np.array(df_v["X [V]"] + 1j * df_v["Y [V]"])
    curr = np.array(df_i["X [V]"] + 1j * df_i["Y [V]"])

    sens_v = 2 * df_v["Escala [V]"] / 2 ** 16
    sens_i = 2 * df_i["Escala [V]"] / 2 ** 16

    err_volt = np.sqrt(df_v["Ruido X [V]"] ** 2 + sens_v ** 2) + \
        1j * np.sqrt(df_v["Ruido Y [V]"] ** 2 + sens_v ** 2)

    err_curr = np.sqrt(df_i["Ruido X [V]"] ** 2 + sens_i ** 2) + \
        1j * np.sqrt(df_i["Ruido Y [V]"] ** 2 + sens_i ** 2)

    phase = np.arctan2(
        np.imag(volt), np.real(volt)
    ) - np.arctan2(
        np.imag(curr), np.real(curr)
    )

    chi = - np.abs(volt) / (freq * np.abs(curr)) * np.exp(1j * phase)

    chi_err = error_chi(freq, curr, volt, err_curr, err_volt, phase)

    return np.asarray(chi, dtype=complex), np.asarray(chi_err, dtype=complex)

# %%


lockin = SR830("GPIB0::8::INSTR")

print(f"Conectado al Lock-in: {lockin.query('*IDN?')}")

# %%

Q = 100  # Factor de calidad
FILTER_SLOPE = 12
AMPLITUDE = 1  # volts

FREQ_MIN = 3e1
FREQ_MAX = 2e3
N_SAMPLES = 500

freq = np.linspace(FREQ_MIN, FREQ_MAX, N_SAMPLES)
lockin.set_internal_ref_cfg(voltage=AMPLITUDE)
lockin.set_use_internal_ref(True)
lockin.set_filter_cfg(slope=FILTER_SLOPE)

# %%

MATERIAL = "cobre"
GEOMETRIA = "grande"

# Voltaje (A-B)

lockin.set_input("A-B")
df_v = measure(lockin, freq, Q=Q)

filepath = PATH / f"con {MATERIAL} {GEOMETRIA} A-B.csv"
df_i.to_csv(filapath)
print(f"Se guardó {filepath}")

plot_measure(df_i)

# %%

# Corriente (A)

lockin.set_input("A")
df_i = measure(lockin, freq, Q=Q)

filepath = PATH / f"con {MATERIAL} {GEOMETRIA} I.csv"
df_i.to_csv(filepath)
print(f"Se guardó {filepath}")

plot_measure(df_i)

# %%

chi, chi_err = calculate_chi(df_v, df_i)

fig, ax = plt.subplots(1, 1)

ax.errorbar(
    freq,
    chi.real,
    yerr=chi_err.real,
    label="Parte real"
)
ax.errorbar(
    freq,
    chi.imag,
    yerr=chi_err.imag,
    label="Parte imaginaria"
)
ax.set(
    xlabel="Frecuencia [Hz]",
    ylabel="Susceptibilidad",
)
ax.legend()

plt.show()

# %%


def chi2_r(
    residue,
    yerr,
    n_data,
    n_params
):
    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return chi2 / degrees_of_freedom


def p_value(
    residue,
    yerr,
    n_data,
    n_params
):
    chi2 = np.sum((residue / yerr) ** 2)
    degrees_of_freedom = n_data - n_params

    return scipy.stats.chi2.sf(chi2, df=degrees_of_freedom)


def fit(model, freq, chi, chi_err, p0):
    try:
        p_opt_real, p_cov_real = curve_fit(
            lambda f, k, alpha, scale, offset: np.real(
                model_func(f, k, alpha, scale, offset)
            ),
            freq,
            chi.real,
            p0=p0,
            sigma=chi_err,
            absolute_sigma=True
        )

    except Exception:
        print("No se pudo ajustar la parte real.")
        p_opt_real, p_cov_real = None, None

    try:
        p_opt_imag, p_cov_imag = curve_fit(
            lambda f, k, alpha, scale, offset: np.imag(
                model_func(f, k, alpha, scale, offset)
            ),
            freq,
            chi.imag,
            p0=p0,
            sigma=chi_err,
            absolute_sigma=True
        )

    except Exception:
        print("No se pudo ajustar la parte imaginaria.")
        p_opt_imag, p_cov_imag = None, None

    if p_opt_real is None and p_opt_imag is None:
        raise Exception

    p_err_real = np.sqrt(np.diag(p_cov_real))
    p_err_imag = np.sqrt(np.diag(p_cov_imag))

    return p_opt_real, p_err_real, p_opt_imag, p_err_imag


def model(f, k, alpha, scale, offset):
    f = np.array(f)
    x = (-1 + 1j) * np.sqrt(np.pi * MU0 * np.abs(k) * f)

    return - scale * jv(2, x) / jv(0, x) * np.exp(1j * np.pi / 180 * alpha) + offset

# %%


fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[3, 1, 1])

K0 = RADIO ** 2 / RHO
CHI_MAX = (chi.real ** 2 + chi.imag ** 2).max()
p0 = [K0, 0, CHI_MAX, 0]

p_opt_real, p_err_real, p_opt_imag, p_err_imag = fit(
    model,
    freq,
    chi,
    chi_err,
    p0
)

chi_fit = np.real(model(freq, *p_opt_real)) + 1j * np.imag(model(freq), *p_err_imag)

k0 = [p_opt_real[0], p_opt_imag[0]]
k0_err = [p_err_real[0], p_err_imag[0]]

residuos = [
    chi.real - chi_fit.real,
    chi.imag - chi_fit.imag,
]

chi2 = [
    chi2_r(residuos[0], chi_err.real, len(residuos[0]), len(p0)),
    chi2_r(residuos[1], chi_err.imag, len(residuos[1]), len(p0)),
]

print(f"Se esperaba obtener k0 = {K0:.0e}.")
print(f"Se obtuvo (real) {k0[0]:.0e} +- {k0_err[0]:.0e}")
print(f"Se obtuvo (imag) {k0[1]:.0e} +- {k0_err[1]:.0e}")
print("")
print(f"chi^2_red = {0:.2e} (real), {0:.2e} (imag)")
print(f"P valor = {0:.2e} (real), {0:.2e} (imag)")


fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[3, 1, 1])

ax[0].set(
    ylabel="Susceptibilidad",
)
ax[1].set(
    ylabel="Residuos",
)
ax[2].set(
    xlabel="Frecuencia [Hz]",
    ylabel="Residuos",
)

ax[0].errorbar(
    freq,
    chi.real,
    yerr=chi_err.real,
    label="Parte real"
)
ax[0].errorbar(
    freq,
    chi.imag,
    yerr=chi_err.imag,
    label="Parte imaginaria"
)

ax[0].plot(
    freq,
    chi_fit.real,
    label="Ajuste real"
)
ax[0].plot(
    freq,
    chi_fit.imag,
    label="Ajuste imaginario"
)

ax[1].axhline(
    y=0,
    color="black",
    alpha=0.9
)
ax[1].errorbar(
    freq,
    residuos[0],
    label="Residuos parte real"
)

ax[2].axhline(
    y=0,
    color="black",
    alpha=0.9
)
ax[2].errorbar(
    freq,
    residuos[1],
    label="Residuos parte imaginaria"
)

for axis in ax:
    axis.legend()

plt.show()
