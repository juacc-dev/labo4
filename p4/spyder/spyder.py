import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
import logging
import re
import time
import datetime
from enum import Enum
from pathlib import Path

PATH = Path(r"C:\Users\publico\Documents\L4 G6 2025-2\data\2 - 19-11")

# %%

count = 2

# %%

logger = logging.getLogger("MAIN")


def print_devices() -> None:
    rm = pyvisa.ResourceManager()

    ids = rm.list_resources()

    for id in ids:
        print(id)


def print_dev_match(interface) -> None:
    rm = pyvisa.ResourceManager()

    ids = rm.list_resources()

    for id in ids:
        if re.match(f"{interface}.*INSTR", id):
            try:
                dev = rm.open_resource(id)
            except Exception:
                print("Timeout")
            print(f"{id}  -->  {dev.query('*IDN?')}")


class VisaInstrument:
    # Possible backends are NI-VISA (empty string) and PyVISA-Py ("@py")
    backend = ""  # NI-VISA
    startup_sleep = 0.5

    def __init__(
        self,
        address,
        **kwargs
    ) -> None:
        rm = pyvisa.ResourceManager(VisaInstrument.backend)

        self._instrument = rm.open_resource(
            resource_name=address,
            # read_termination='\n',
            # write_termination='\n',
            **kwargs
        )

        time.sleep(VisaInstrument.startup_sleep)
        self.check()

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

    def write(
        self,
        cmd: str,
        **kwargs
    ) -> None:
        """Write a single value to the instrument."""

        return self._instrument.write(cmd, **kwargs)

    def query(
        self,
        cmd: str,
        **kwargs
    ) -> str:
        """Query a single value from the instrument."""

        return self._instrument.query(cmd, **kwargs)

    def write_values(
        self,
        cmd: str,
        *,
        ascii: bool = True,
        **kwargs
    ):
        """Write multiple values to the instrument. By default, the messsage is
        sent in binary form, but passing ascii=True makes it use plain text."""

        if ascii:
            return self._instrument.write_ascii_values(
                cmd,
                **kwargs
            )

        else:
            return self._instrument.write_binary_values(
                cmd,
                **kwargs
            )

    def query_values(
        self,
        cmd: str,
        *,
        ascii: bool = True,
        separator=",",
        **kwargs
    ):
        """Query multiple values from the instrument. By default, the result is
        sent in binary form, but passing ascii=True makes it use plain text."""

        if ascii:
            return self._instrument.query_ascii_values(
                cmd,
                separator=separator,
                **kwargs
            )

        else:
            return self._instrument.query_binary_values(
                cmd,
                **kwargs
            )

    def clear(self):
        return self.write("*CLS")

    def opc(self) -> bool:
        return self.query("*OPC?") == "1"


# %%


class Tektronix_AFG3021B(VisaInstrument):
    """Tektronix AFG3021 function generator."""

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
        offset=None,
        freq=None,
        shape: Funs | str = None,
        impedance=None
    ):
        """Set configuarion for the function generator."""

        if voltage is not None:
            self.write(f"SOURce{ch}:VOLTage {voltage}")

        if offset is not None:
            self.write(f"SOURce{ch}:VOLTage:OFFSET {offset}")
    
        if freq is not None:
            self.write(f"SOURce{ch}:FREQuency {freq}")

        if shape is not None:
            shape = shape.value if type(
                shape) is Tektronix_AFG3021B.Funs else shape

            self.write(f"SOURce{ch}:FUNCtion:SHAPe {shape}")

        if impedance is not None:
            self.write(f"OUTPut{ch}:IMPedance {impedance}")

    def get_voltage(self):
        return float(self.query("SOURce1:VOLTage?"))

    def get_freq(self):
        return float(self.query("SOURce1:FREQuency?"))

    def get_shape(self):
        return self.query("SOURce1:FUNCtion:SHAPe?")

    def get_impedance(self):
        return self.query("OUTPut1:IMPedance?")


# %%


def tuplify(param) -> tuple:
    if not isinstance(param, tuple):
        return [param, param]

    return param


class Tektronix_TDS1002B(VisaInstrument):
    """Tektronix TDS1002B oscilloscope."""

    X_ACCURACY = 50.0 / 10 ** 6  # 50 ppm
    Y_ACCURACY = 0.03  # 3% of measurement
    Y_DIVISIONS = 8
    X_DIVISIONS = 10
    SCREEN_HEIGHT = 255
    SCREEN_WIDTH = 2500
    BAUD_RATE = 19200  # bits per second

    LINE_TERMINATOR_STR = "LF"
    LINE_TERMINATOR_CODE = '\n'

    DATA_ENCODING = "RPBinary"  # positive integer, from 0 to 255
    DATA_WIDTH = 1  # 8 bits

    def __init__(
        self,
        address,
        **kwargs
    ):
        super().__init__(address, **kwargs)

        self.write(f"RS232:BAUd {Tektronix_TDS1002B.BAUD_RATE}")
        self._instrument.baud_rate = Tektronix_TDS1002B.BAUD_RATE

        self.write(f"RS232:TRANsmit:TERMinator {
                   Tektronix_TDS1002B.LINE_TERMINATOR_STR}")
        self._instrument.read_terminator = Tektronix_TDS1002B.LINE_TERMINATOR_CODE
        self._instrument.write_terminator = Tektronix_TDS1002B.LINE_TERMINATOR_CODE

    def acquire(
        self,
        *,
        on: bool = None,
        avg: int = None  # Only supports 4, 16, 64 and 128
    ) -> None:
        if avg is not None:
            self.write(f"ACQuire:MODe AVErage {avg}")

        if on is not None:
            state = 1 if on is True else 0
            self.write(f"ACQuire:STATE {state}")

    def set(
        self,
        ch,
        *,
        height: float = None,  # In volts
        y0: float = None,      # In volts
        width: float = None,   # In seconds
        x0: float = None       # In seconds
    ) -> None:
        """Set configuration for the oscilloscope."""

        if height is not None:
            scale = height / Tektronix_TDS1002B.Y_DIVISIONS

            self.write(f"CH{ch}:SCAle {scale:.1E}")

        if y0 is not None:
            scale = self.query(f"CH{ch}:SCAle?")
            pos = y0 / scale

            # The position shuold be in divisions (of the screen)
            self.write(f"CH{ch}:POSition {pos:.1E}")

        if width is not None:
            scale = width / Tektronix_TDS1002B.X_DIVISIONS

            self.write(f"HORizontal:SCAle {scale:.1E}")

        if x0 is not None:
            scale = self.query("HORizontal:SCAle?")
            pos = x0 / scale

            self.write(f"HORizontal:POSition {pos:.1E}")

    def get(
        self,
        ch,
        parameter: str
    ):
        """Query configuration of the oscilloscope."""

        match parameter:
            case "height":
                scale = self.query("CH{ch}:SCAle?")
                height = scale * Tektronix_TDS1002B.Y_DIVISIONS

                return height

            case "y0":
                scale, pos = self.query_values(
                    "CH{ch}:SCAle?;POSition?",
                    ascii=True,
                    separator=';'
                )
                y0 = scale * pos

                return y0

            case "width":
                scale = self.query("HORizontal:SCAle?")
                width = scale * Tektronix_TDS1002B.X_DIVISIONS

                return width

            case "x0":
                scale, pos = self.query_values(
                    "HORizontal:SCAle;POSition?",
                    ascii=True,
                    separator=';'
                )
                x0 = scale * pos

                return x0

    def curve(
        self,
        ch,
    ) -> pd.DataFrame:
        """Retrieve curve data (the oscilloscope's screen) for a sigle channel.
        Returns a dataframe containing time and voltage, each with their
        uncertainty. The structure is
        ```csv
        Tiempo [s], Error X [s], Voltage [V], Error Y [V]
        ```"""

        # Vertical units are the units of the Y axis in the oscilloscope's
        # screen, usually volts.
        # Horizontal units are the units of the X axis, usually seconds.

        # Set data source, the binary encoding and the byte width of a point.
        self.write(
            f"DATa:SOURce CH{ch};ENCdg {Tektronix_TDS1002B.DATA_ENCODING};WIDth {
                Tektronix_TDS1002B.DATA_WIDTH}"
        )

        # TODO: Maybe wait now?

        # Get parameters for converting the raw curve data into voltage.
        x0, dx, y0, vertical_units, vertical_offset = self.query_values(
            "WFMPre:XZEro?;XINcr?;YZEro?;YMUlt?;YOFf?",
            ascii=True,  # The response is in plain text.
            separator=';'
        )
        # Here's what they mean:
        # - x0: Horizontal adjustment (controlled by a knob).
        # - dx: Step size. Pixel width in horizontal units.
        # - y0: Vertical adjustment (controlled by a knob).
        # - vertical_offset: adjust raw_data for converting units. It's
        #   probably 127.
        # - vertical_units: converting factor, something like volts/pixel
        #   (in the manual it's 'YUNits per digitizer level').

        # Retrieve raw curve data, in bytes. The values range from 0 to 255.
        raw_data = self.query_values(
            "CURVe?",
            datatype="B",  # bytes
            container=np.array,
            is_big_endian=False  # Little endian
        )

        # TODO: maybe check if it's alright?

        logger.info(f"Read {raw_data.size} points from oscilloscope.")

        # Actual, meaningful curve data, in the corresponing units (like vols).
        y = y0 + (raw_data - vertical_offset) * vertical_units

        # Horizontal axis, in the correspoing units (like seconds).
        x = np.arange(x0, x0 + Tektronix_TDS1002B.SCREEN_WIDTH, dx)

        # Size of a pixel in vertical units (minimum unit of measurement).
        sensitivity = self.query(
            f"CH{ch}:SCAle?") * Tektronix_TDS1002B.Y_DIVISIONS / Tektronix_TDS1002B.SCREEN_HEIGHT

        # Uncertainty for each point, from the accuracy of the measurement
        # and the sensitivity of the Instrument.
        yerr = Tektronix_TDS1002B.Y_ACCURACY * (y - y0) + sensitivity

        # The horizontal uncertainty is always the same.
        xerr = Tektronix_TDS1002B.X_ACCURACY

        df = pd.DataFrame({
            "Tiempo [s]": pd.Series(x),
            "Error X [s]": pd.Series(xerr),
            "Voltaje [V]": pd.Series(y),
            "Error Y [V]": pd.Series(yerr)
        })

        return df

    def curve2(self):
        """Retrieve curve data from both channels of the oscilloscope.
        Returns a dataframe containing time, voltage for each channel and their
        uncertainty.
        ```csv
        Tiempo [s], Error X [s], Canal 1 [V], Error 1 [V], Canal 2 [V], Error 2
        [V]
        ```"""

        channel_1 = self.curve(1)
        channel_2 = self.curve(2)

        x = channel_1[channel_1.columns[0]]
        xerr = channel_1[channel_1.columns[1]]

        ch1 = channel_1[channel_1.columns[2]]
        ch1_err = channel_1[channel_1.columns[3]]

        ch2 = channel_2[channel_2.columns[2]]
        ch2_err = channel_2[channel_2.columns[3]]

        df = pd.DataFrame({
            "Tiempo [s]": x,
            "Error X [s]": xerr,
            "Canal 1 [V]": ch1,
            "Error 1 [V]": ch1_err,
            "Canal 2 [V]": ch2,
            "Error 2 [V]": ch2_err
        })

        return df


# %%

class Agilent_34970A(VisaInstrument):
    """Agilent 34970A multiplexor.
    Agilent's former Test and Measurement business has become Keysight
    Technologies."""

    MISTERY_DELAY = 0.1
    MISTERY_DELAY_EXTRA = 0.5

    class Vars(Enum):
        CURRENT_AC = "CURRENT:AC"
        CURRENT_DC = "CURRENT:DC"
        BYTE = "DIGITAL:BYTE"
        FREQUENCY = "FREQUENCY"
        RESISTANCE = "RESISTANCE"
        FRESISTANCE = "FRESISTANCE"
        PERIOD = "PERIOD"
        TEMPERATURE = "TEMPERATURE"
        TOTALIZE = "TOTALIZE"
        VOLTAGE_AC = "VOLTAGE:AC"
        VOLTAGE_DC = "VOLTAGE:DC"

    class Channel:
        def __init__(
            self,
            channel: int,
            *,
            variable=None,
            conf_args=None,
            delay: float = None,
        ):
            self.channel = channel
            self.variable = variable
            self.conf_args = conf_args
            self.delay = delay

    def __init__(
        self,
        address,
        **kwargs
    ):
        super().__init__(address, **kwargs)

        self.ch_list = {}


    def channel_str(self):
        channels = self.ch_list.values()

        return str([ch.channel for ch in channels])[1:-1]

    def total_delay(self):
        delay = 0.0

        for ch in self.ch_list.values():
            delay += ch.delay + Agilent_34970A.MISTERY_DELAY

        return delay + Agilent_34970A.MISTERY_DELAY_EXTRA

    def config_channel(
        self,
        ch: Channel,
    ):
        self.ch_list[ch.channel] = ch

        if ch.variable is not None:
            var = ch.variable.value if type(
                ch.variable) is Agilent_34970A.Vars else ""

            if ch.conf_args is None:
                self.write(f"CONFIGURE:{var}")
            else:
                self.write(f"CONFIGURE:{var} {ch.conf_args},(@{ch.channel})")

        if ch.delay is not None:
            self.write(f"ROUTE:CHANNEL:DELAY {ch.delay},(@{ch.channel})")

    def config(
        self,
        channels: list[Channel],
        *,
        trigger_interval: float = None,
        n_sweep: int = 1
    ):
        for ch in channels:
            self.config_channel(ch)

        if trigger_interval is not None:
            self.write(f"TRIGGER:TIMER {trigger_interval}")
            self.write(f"TRIGGER:COUNT {n_sweep}")

        self.write(f"ROUTE:SCAN (@{self.channel_str()})")
        
        self.write("FORMAT:READING:CHANNEL ON")
        self.write("FORMAT:READING:TIME ON")
        self.write("FORMAT:READING:UNIT OFF")
        self.write("FORMAT:READING:ALARM ON")

        self.write("FORMAT:READING:TIME:TYPE ABSOLUTE")

    def parse_time(crap):
        time = [
            datetime.datetime(
                int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]),
                int(x[5]), int((x[5] % 1)*1000000)
            ).timestamp()
            for x in np.transpose(crap)
        ]

        return np.array(time)

    def parse_read(self, data_raw):
        n_values = 9
        data = np.transpose(
            np.reshape(
                np.array(data_raw),
                (len(self.ch_list.keys()), n_values)
            )
        )

        measurement = data[0]

        time = Agilent_34970A.parse_time(data[1:7])

        channel = data[7]

        alert = data[8]

        return measurement, time, channel, alert

    def scan(
        self,
    ):
        data_raw = self.query_values("READ?")

        value, time, channel, alert = self.parse_read(data_raw)

        dfs = {}

        for ch, t, val, al in zip(channel, time, value, alert):
            df = pd.DataFrame({
                "Time": [t],
                "Value": [val],
                "Alert": [al]
            })

            dfs[int(ch)] = df

        return dfs


# %%

def measure(
    mux,
    *,
    time_total,
    time_step
):
    min_step = mux.total_delay()

    if time_step < min_step:
        time_step = min_step
        logger.warning(f"Usando time_step = {min_step:.2f} s")

    n_points = int(time_total / time_step)

    eta = datetime.datetime.fromtimestamp(
        n_points * time_step + time.time()
    )

    print(f"Total de puntos a medir: {n_points} (dt = {time_step:.2f} s)")
    print(f"Se espera que finalice: {eta.hour}:{eta.minute} hs")

    dfs = {}
    t0 = time.time()
    t = time.time() - t0

    for i in range(n_points):
        t0 = time.time()

        dfs_tmp = mux.scan()
        
        for ch, df in dfs_tmp.items():
            if ch not in dfs.keys():
                dfs[ch] = df

            else:
                dfs[ch] = pd.concat([dfs[ch], df], ignore_index=True)

        t = time.time() - t0
        sleep = time_step - t
        time.sleep(sleep if sleep > 0 else 0)
                
        print(f"\x1b[2K{100 * (i+1) / n_points:.2f} %", end="")

    return dfs


def plot_df(df, label=None):
    t = df["Time"]
    temp = df["Value"]
    
    plt.plot(t, temp, "o", label=label)



# %%

Vars = Agilent_34970A.Vars
Ch = Agilent_34970A.Channel

mux = Agilent_34970A("ASRL3::INSTR")
fungen = Tektronix_AFG3021B("USB0::0x0699::0x0346::C034165::INSTR")

print(f"Conectado al multiplexor: {mux.query('*IDN?')}")
print(f"Conectado al generador: {fungen.query('*IDN?')}")

# %%

CHANNEL_DELAY = 0.2
TIME_TOTAL = 45 * 60
TIME_STEP = 5

K_COUPLE = {
    "variable": Vars.TEMPERATURE,
    "conf_args": "TC,K",
    "delay": CHANNEL_DELAY,
}

J_COUPLE = {
    "variable": Vars.TEMPERATURE,
    "conf_args": "TC,J",
    "delay": CHANNEL_DELAY,
}

mux.clear()

mux.config([
    Ch(101, **K_COUPLE),
    Ch(102, **K_COUPLE),
    Ch(103, **J_COUPLE),
    Ch(104, **J_COUPLE),
    Ch(105, **J_COUPLE),
    Ch(107, **J_COUPLE),
])

fungen.set(
    freq=200,
    voltage=5,
    offset=2.5
)

fungen.write("SOURCE1:PWM:INTERNAL:FREQ 0.005")
fungen.write("SOURCE1:PWM:INTERNAL:FUNCTION SIN")
fungen.write("SOURCE1:PWM:DEVIATION:DCYCLE 50.0")

fungen.write("SOURCE1:PWM:STATE ON")

fungen.output("OFF")

# %%

dfs = measure(
    mux,
    time_total=TIME_TOTAL,
    time_step=TIME_STEP
)

path = PATH / f"{count} (enfriado)"
path.mkdir( parents=True, exist_ok=True)

for ch, df in dfs.items():
    filename = f"canal {ch}.csv"
    df.to_csv(path / filename)

    print(f"Se guardó {filename} en {path.stem}")

count += 1
#%%
for ch, df in dfs.items():
   plot_df(df, label=str(ch))
   plt.legend()

plt.show()

# %%
