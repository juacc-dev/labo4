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


def promedio(lockin, n=30, tiempo=0):
    z = np.zeros(n, dtype=complex)

    for i in range(n):
        z[i] = lockin.right_value()
        time.sleep(tiempo / n)

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

        # Mantiene el factor de calidad ~constante
        integration_time = Q / freq
        integration_time = lockin.set_filter_cfg(time=integration_time)[0]

        time.sleep((N_TAU - 1) * integration_time)

        data[i], noise[i] = promedio(lockin, n=30, tiempo=1)

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
