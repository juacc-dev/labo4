import pandas as pd
import numpy as np
import nidaqmx
import time
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration, \
    READ_ALL_AVAILABLE


def list_devices():
    system = nidaqmx.system.System.local()

    return system.devices


def measure(
    channel_name: str,
    duration,           # in seconds
    sampling_freq: int  # in hertz
) -> pd.DataFrame:
    n_samples = int(duration * sampling_freq)
    duration = n_samples * sampling_freq

    with nidaqmx.Task() as task:
        # Set analog channel
        task.ai_channels.add_ai_voltage_chan(
            channel_name,
            terminal_config=TerminalConfiguration.DIFF
        )

        # Set sampling configuration
        task.ai_channels.cfg_samp_clk_timing(
            sampling_freq,
            samps_per_chan=n_samples,
            sample_mode=AcquisitionType.FINITE
        )

        data = task.read(
            number_of_samples_per_channel=READ_ALL_AVAILABLE,
            # timeout=duration+0.1
        )

    t = np.linspace(0, duration, n_samples)
    y = np.asarray(data)

    df = pd.DataFrame({
        "Tiempo [s]": pd.Series(t),
        "Error T [s]": pd.Series(),
        "Voltaje [V]": pd.Series(y),
        "Erorr T [V]": pd.Series()
    })

    return df


# Split measurement in multiple reads
# def measure_chained(
#     channel_name: str,
#     duration,
#     sampling_freq: int,
#     delay: float,
# ):
#     n_samples = int(duration * sampling_freq)
#     duration = n_samples * sampling_freq

#     with nidaqmx.Task() as task:
#         task.ai_channels.add_ai_voltage_chan(
#             channel_name,
#             terminal_config=TerminalConfiguration.DIFF
#         )

#         task.timing.cfg_samp_clk_timing(
#             sampling_freq,
#             sample_mode=AcquisitionType.CONTINUOUS
#         )

#         task.start()
#         t0 = time.time()

#         times = []
#         data = []  # list of arrays

#         n = 0
#         while n < n_samples:
#             time.sleep(delay)

#             new_data = task.read(
#                 number_of_samples_per_channel=READ_ALL_AVAILABLE
#             )

#             t = time.time() - t0

#             n += len(new_data)

#             times.insert(t)
#             data.insert(np.asarray(new_data))

#         return t, data
