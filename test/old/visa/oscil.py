# import time
import numpy as np
import pandas as pd
import logging
from visa import VisaInstrument

logger = logging.getLogger("visa.oscil")

X_ACCURACY = 50.0 / 10 ** 6  # 50 ppm
Y_ACCURACY = 0.03  # 3% of measurement
Y_DIVISIONS = 8
X_DIVISIONS = 10
SCREEN_HEIGHT = 255
SCREEN_WIDTH = 2500

DATA_ENCODING = "RPBinary"  # positive integer, from 0 to 255
DATA_WIDTH = 1  # 8 bits

BAUD_RATE = 19200  # bits per second

LINE_TERMINATOR_STR = "LF"
LINE_TERMINATOR_CODE = '\n'

BUSY_DELAY = 0.1  # seconds
BUSY_MAX_TRIES = 10


def tuplify(param) -> tuple:
    if not isinstance(param, tuple):
        return [param, param]

    return param


class Oscilloscope(VisaInstrument):
    def __init__(
        self,
        address,
        backend: str = None,
        **kwargs
    ):
        super().__init__(address, backend=backend, **kwargs)

        self.write(f"RS232:BAUd {BAUD_RATE}")
        self._instrument.baud_rate = BAUD_RATE

        self.write(f"RS232:TRANsmit:TERMinator {LINE_TERMINATOR_STR}")
        self._instrument.read_terminator = LINE_TERMINATOR_CODE
        self._instrument.write_terminator = LINE_TERMINATOR_CODE

    # def wait_busy(
    #     self,
    #     delay=BUSY_DELAY,
    #     max_tries=BUSY_MAX_TRIES
    # ):
    #     n = 0

    #     while self.query("BUSY?") or n > max_tries:
    #         time.sleep(delay)

    #     if n > max_tries:
    #         logger.error(f"Wait timed out! (>{max_tries * delay * 1000} ms)")

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
            scale = height / Y_DIVISIONS

            self.write(f"CH{ch}:SCAle {scale:.1E}")

        if y0 is not None:
            scale = self.query(f"CH{ch}:SCAle?")
            pos = y0 / scale

            # The position shuold be in divisions (of the screen)
            self.write(f"CH{ch}:POSition {pos:.1E}")

        if width is not None:
            scale = width / X_DIVISIONS

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
                height = scale * Y_DIVISIONS

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
                width = scale * X_DIVISIONS

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
            f"DATa:SOURce CH{ch};ENCdg {DATA_ENCODING};WIDth {DATA_WIDTH}"
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
        x = np.arange(x0, x0 + SCREEN_WIDTH, dx)

        # Size of a pixel in vertical units (minimum unit of measurement).
        sensitivity = self.query(
            f"CH{ch}:SCAle?") * Y_DIVISIONS / SCREEN_HEIGHT

        # Uncertainty for each point, from the accuracy of the measurement
        # and the sensitivity of the Instrument.
        yerr = Y_ACCURACY * (y - y0) + sensitivity

        # The horizontal uncertainty is always the same.
        xerr = X_ACCURACY

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
