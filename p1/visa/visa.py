# Information about SCPI:
# https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments
# SCPI Volume 1: Syntax and style (1999):
# https://www.ivifoundation.org/downloads/SCPI/scpi-99.pdf

# User guide for PyVISA:
# https://pyvisa.readthedocs.io/en/latest/introduction/index.html

# TODO: Check if the instruments actually support *OPC

import pyvisa
import logging
import re

logger = logging.getLogger("visa")

# Possible backends are NI-VISA (default) and PyVISA-Py ("@py")
DEFAULT_BACKEND = ""  # NI-VISA


def list_usb() -> dict[str, str]:
    """Return a dict mapping instrument's address to human-readable name for
    each available USB instrument."""

    rm = pyvisa.ResourceManager()

    ids = rm.list_resources()

    usb_devices = {}

    for id in ids:
        if re.match(r"USB::.*::INSTR", id):
            dev = rm.open_resource(id)

            usb_devices[id] = dev.query("*IDN?")

    return usb_devices


# def channel_list(ch: int) -> list[int]:
#     ch_list = [1, 2]

#     match ch:
#         case 0:
#             pass
#         case 1:
#             ch_list = [1]
#         case 2:
#             ch_list = [2]
#         case _:
#             logger.error(f"Invalid channel: {ch}")
#             return None

#     return ch_list


class VisaInstrument:
    def __init__(
        self,
        address,
        *,
        backend: str = DEFAULT_BACKEND,
        **kwargs
    ) -> None:

        self._instrument = pyvisa.ResourceManager(backend).open_resource(
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
